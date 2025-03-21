import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
from PIL import Image
import os
#from os import join
from time import time
import numpy as np
import cv2 
import torchfile

from ultralytics import YOLO
import ultralytics.engine.results
from math import floor, ceil, sqrt
from .utils import plot_3D_pts, plot_bboxes

from .models import CreateModel_pascal


# Mean Car Shape
S = torch.Tensor([[-0.8606, -1.3216, -0.6533],
                  [-0.7902,  1.1585, -0.5346],
                  [ 0.7291, -1.2980, -0.6527],
                  [ 0.6426,  1.1804, -0.5486],
                  [-0.5758, -0.3267,  0.8473],
                  [ 0.4569, -0.3152,  0.8475],
                  [ 0.4217,  0.8741,  0.8666],
                  [-0.5511,  0.8756,  0.8675],
                  [-0.6317, -2.0092, -0.1013],
                  [ 0.5008, -1.9784, -0.1010],
                  [ 0.4336,  1.8508,  0.0929],
                  [-0.5968,  1.8353,  0.0974]])


"""
Convert axis-angle representation to a 3x3 rotation matrix
"""
class Rodrigues(torch.autograd.Function):
    @staticmethod
    def forward(self, inp):
        pose = inp.detach().cpu().numpy()
        rotm, part_jacob = cv2.Rodrigues(pose)
        self.jacob = torch.Tensor(np.transpose(part_jacob)).contiguous()
        rotation_matrix = torch.Tensor(rotm.ravel())
        return rotation_matrix.view(3,3)

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.view(1,-1)
        grad_input = torch.mm(grad_output, self.jacob)
        grad_input = grad_input.view(-1)
        return grad_input

rodrigues = Rodrigues.apply

class PoseEstimation(object):

    def __init__(self):

        self.R_reg_2_opt = torch.Tensor([[0,-1,0],
                                [0,0,-1],
                                [1,0,0]])

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_keypoints_file_path = "./aux_data/keypoint_description/pascal_num_keypoints.txt"
        self.yolo = YOLO("./yolo_models/yolo11x-seg.pt")

        ## Camera K parameters
        self.cam_fx= 718.856
        self.cam_fy= 718.856
        self.cam_cx= 607.1928
        self.cam_cy= 185.2157

        ## Parameters for Keypoint detections and image resizing
        self.img_size = 256 
        self.activation_threshold = 0.1


        self.keypoints_indices = dict()
        start_index = 0
        with open(self.num_keypoints_file_path, 'r') as num_keypoints_file:
            for line in num_keypoints_file:
                split = line.split(' ')
                if len(split) == 2:
                    self.keypoints_indices[split[0]] = \
                        (start_index, start_index + int(split[1]))
                    start_index += int(split[1])
        print("Keypoint indices:", self.keypoints_indices)

        self.K = np.array([[self.cam_fx, 0, self.cam_cx],
                    [0,self.cam_fy, self.cam_cy],
                    [0,0,1]])
        self.K = torch.from_numpy(self.K).float().to('cpu')

        print("K Matrix")
        print(self.K)

        self.model = self.convert_model('./aux_data/keypoint_models/pose-hg-pascal3d.t7') # Called from within the "get_keypoints" func
        self.model.cuda()
        self.model.eval()

    def convert_model(self, torch_model):

        model = CreateModel_pascal()
        
        model_lua = torchfile.load(torch_model)
        
        # conv
        layer_conv = ['model.layer1',
        'model.layer4.layer2.layer1',
        'model.layer8.layer2.layer1',
        'model.layer9.layer3.layer2.layer1',
        'model.layer9.layer8.layer3.layer2.layer1',
        'model.layer9.layer8.layer8.layer3.layer2.layer1',
        'model.layer9.layer8.layer8.layer8.layer3.layer2.layer1',
        'model.layer9.layer8.layer8.layer8.layer8.layer2.layer1',
        'model.layer10.layer1',
        'model.layer11.layer1',
        'model.layer12',
        'model.layer13',
        'model.layer14',
        'model.layer15.layer1.layer2.layer1',
        'model.layer15.layer3.layer2.layer1',
        'model.layer15.layer5.layer2.layer1',
        'model.layer15.layer8.layer3.layer2.layer1',
        'model.layer15.layer8.layer8.layer3.layer2.layer1',
        'model.layer15.layer8.layer8.layer8.layer3.layer2.layer1',
        'model.layer15.layer8.layer8.layer8.layer8.layer2.layer1',
        'model.layer16.layer1',
        'model.layer17.layer1',
        'model.layer18']
        
        module_conv = [[1],
        [4, 0, 1, 0],
        [8, 0, 1, 0],
        [11, 0, 1, 0],
        [18, 0, 1, 0],
        [25, 0, 1, 0],
        [32, 0, 1, 0],
        [37, 0, 1, 0],
        [50],
        [53],
        [56],
        [59],
        [58],
        [61, 0, 1, 0],
        [63, 0, 1, 0],
        [65, 0, 1, 0],
        [70, 0, 1, 0],
        [77, 0, 1, 0],
        [84, 0, 1, 0],
        [89, 0, 1, 0],
        [102],
        [105],
        [108]]
        
        
        for layer, module in zip(layer_conv, module_conv):
            module_string = 'model_lua'
            for module_id in module:
                module_string = '%s[\'modules\'][%d]' % (module_string, module_id)
            
            exec_string = '%s.weight.data = torch.FloatTensor(%s[\'weight\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.bias.data = torch.FloatTensor(%s[\'bias\'])' %(layer, module_string); exec(exec_string)
        
        
        # batchnorm
        layer_batchnorm = ['model.layer2',
        'model.layer10.layer2',
        'model.layer11.layer2',
        'model.layer16.layer2',
        'model.layer17.layer2']
        
        
        module_batchnorm = [[2],
        [51],
        [54],
        [103],
        [106]]
        
        
        for layer, module in zip(layer_batchnorm, module_batchnorm):
            module_string = 'model_lua'
            for module_id in module:
                module_string = '%s[\'modules\'][%d]' % (module_string, module_id)
        
            exec_string = '%s.weight.data = torch.FloatTensor(%s[\'weight\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.bias.data = torch.FloatTensor(%s[\'bias\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.running_mean.data = torch.FloatTensor(%s[\'running_mean\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.running_var.data = torch.FloatTensor(%s[\'running_var\'])' %(layer, module_string); exec(exec_string)
        
        
        # Residual
        layer_residual = ['model.layer4.layer1',
        'model.layer6.layer1',
        'model.layer7.layer1',
        'model.layer8.layer1',
        'model.layer9.layer1.layer1',
        'model.layer9.layer2.layer1',
        'model.layer9.layer3.layer1',
        'model.layer9.layer5.layer1',
        'model.layer9.layer6.layer1',
        'model.layer9.layer7.layer1',
        'model.layer9.layer8.layer1.layer1',
        'model.layer9.layer8.layer2.layer1',
        'model.layer9.layer8.layer3.layer1',
        'model.layer9.layer8.layer5.layer1',
        'model.layer9.layer8.layer6.layer1',
        'model.layer9.layer8.layer7.layer1',
        'model.layer9.layer8.layer8.layer1.layer1',
        'model.layer9.layer8.layer8.layer2.layer1',
        'model.layer9.layer8.layer8.layer3.layer1',
        'model.layer9.layer8.layer8.layer5.layer1',
        'model.layer9.layer8.layer8.layer6.layer1',
        'model.layer9.layer8.layer8.layer7.layer1',
        'model.layer9.layer8.layer8.layer8.layer1.layer1',
        'model.layer9.layer8.layer8.layer8.layer2.layer1',
        'model.layer9.layer8.layer8.layer8.layer3.layer1',
        'model.layer9.layer8.layer8.layer8.layer5.layer1',
        'model.layer9.layer8.layer8.layer8.layer6.layer1',
        'model.layer9.layer8.layer8.layer8.layer7.layer1',
        'model.layer9.layer8.layer8.layer8.layer8.layer1',
        'model.layer9.layer8.layer8.layer8.layer9.layer1',
        'model.layer9.layer8.layer8.layer9.layer1',
        'model.layer9.layer8.layer9.layer1',
        'model.layer9.layer9.layer1',
        'model.layer15.layer1.layer1',
        'model.layer15.layer2.layer1',
        'model.layer15.layer3.layer1',
        'model.layer15.layer5.layer1',
        'model.layer15.layer6.layer1',
        'model.layer15.layer7.layer1',
        'model.layer15.layer8.layer1.layer1',
        'model.layer15.layer8.layer2.layer1',
        'model.layer15.layer8.layer3.layer1',
        'model.layer15.layer8.layer5.layer1',
        'model.layer15.layer8.layer6.layer1',
        'model.layer15.layer8.layer7.layer1',
        'model.layer15.layer8.layer8.layer1.layer1',
        'model.layer15.layer8.layer8.layer2.layer1',
        'model.layer15.layer8.layer8.layer3.layer1',
        'model.layer15.layer8.layer8.layer5.layer1',
        'model.layer15.layer8.layer8.layer6.layer1',
        'model.layer15.layer8.layer8.layer7.layer1',
        'model.layer15.layer8.layer8.layer8.layer1.layer1',
        'model.layer15.layer8.layer8.layer8.layer2.layer1',
        'model.layer15.layer8.layer8.layer8.layer3.layer1',
        'model.layer15.layer8.layer8.layer8.layer5.layer1',
        'model.layer15.layer8.layer8.layer8.layer6.layer1',
        'model.layer15.layer8.layer8.layer8.layer7.layer1',
        'model.layer15.layer8.layer8.layer8.layer8.layer1',
        'model.layer15.layer8.layer8.layer8.layer9.layer1',
        'model.layer15.layer8.layer8.layer9.layer1',
        'model.layer15.layer8.layer9.layer1',
        'model.layer15.layer9.layer1']
        
        module_residual = [[4, 0, 0],
        [6, 0, 0],
        [7, 0, 0],
        [8, 0, 0],
        [9, 0, 0],
        [10, 0, 0],
        [11, 0, 0],
        [13, 0, 0],
        [14, 0, 0],
        [15, 0, 0],
        [16, 0, 0],
        [17, 0, 0],
        [18, 0, 0],
        [20, 0, 0],
        [21, 0, 0],
        [22, 0, 0],
        [23, 0, 0],
        [24, 0, 0],
        [25, 0, 0],
        [27, 0, 0],
        [28, 0, 0],
        [29, 0, 0],
        [30, 0, 0],
        [31, 0, 0],
        [32, 0, 0],
        [34, 0, 0],
        [35, 0, 0],
        [36, 0, 0],
        [37, 0, 0],
        [38, 0, 0],
        [41, 0, 0],
        [44, 0, 0],
        [47, 0, 0],
        [61, 0, 0],
        [62, 0, 0],
        [63, 0, 0],
        [65, 0, 0],
        [66, 0, 0],
        [67, 0, 0],
        [68, 0, 0],
        [69, 0, 0],
        [70, 0, 0],
        [72, 0, 0],
        [73, 0, 0],
        [74, 0, 0],
        [75, 0, 0],
        [76, 0, 0],
        [77, 0, 0],
        [79, 0, 0],
        [80, 0, 0],
        [81, 0, 0],
        [82, 0, 0],
        [83, 0, 0],
        [84, 0, 0],
        [86, 0, 0],
        [87, 0, 0],
        [88, 0, 0],
        [89, 0, 0],
        [90, 0, 0],
        [93, 0, 0],
        [96, 0, 0],
        [99, 0, 0],
        ]
        
        
        for layer, module in zip(layer_residual, module_residual):
            module_string = 'model_lua'
            for module_id in module:
                module_string = '%s[\'modules\'][%d]' % (module_string, module_id)
            exec_string = '%s.layer1.weight.data = torch.FloatTensor(%s[\'modules\'][0][\'weight\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.layer1.bias.data = torch.FloatTensor(%s[\'modules\'][0][\'bias\'])' %(layer, module_string); exec(exec_string)
        
            exec_string = '%s.layer1.running_mean.data = torch.FloatTensor(%s[\'modules\'][0][\'running_mean\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.layer1.running_var.data = torch.FloatTensor(%s[\'modules\'][0][\'running_var\'])' %(layer, module_string); exec(exec_string)
        
            exec_string = '%s.layer3.weight.data = torch.FloatTensor(%s[\'modules\'][2][\'weight\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.layer3.bias.data = torch.FloatTensor(%s[\'modules\'][2][\'bias\'])' %(layer, module_string); exec(exec_string)
        
            exec_string = '%s.layer4.weight.data = torch.FloatTensor(%s[\'modules\'][3][\'weight\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.layer4.bias.data = torch.FloatTensor(%s[\'modules\'][3][\'bias\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.layer4.running_mean.data = torch.FloatTensor(%s[\'modules\'][3][\'running_mean\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.layer4.running_var.data = torch.FloatTensor(%s[\'modules\'][3][\'running_var\'])' %(layer, module_string); exec(exec_string)  
        
            exec_string = '%s.layer6.weight.data = torch.FloatTensor(%s[\'modules\'][5][\'weight\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.layer6.bias.data = torch.FloatTensor(%s[\'modules\'][5][\'bias\'])' %(layer, module_string); exec(exec_string)
        
            exec_string = '%s.layer7.weight.data = torch.FloatTensor(%s[\'modules\'][6][\'weight\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.layer7.bias.data = torch.FloatTensor(%s[\'modules\'][6][\'bias\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.layer7.running_mean.data = torch.FloatTensor(%s[\'modules\'][6][\'running_mean\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.layer7.running_var.data = torch.FloatTensor(%s[\'modules\'][6][\'running_var\'])' %(layer, module_string); exec(exec_string)
        
            exec_string = '%s.layer9.weight.data = torch.FloatTensor(%s[\'modules\'][8][\'weight\'])' %(layer, module_string); exec(exec_string)
            exec_string = '%s.layer9.bias.data = torch.FloatTensor(%s[\'modules\'][8][\'bias\'])' %(layer, module_string); exec(exec_string)
        
        return model

    def get_patches_and_bounds(self, detections, image, im_height, im_width):
        ''' Uses the detections to get the patches of the image that contain
        the detected objects.
        '''
        patches = np.zeros((len(detections), 3, self.img_size, self.img_size))
        bounds = []

        for i, detection in enumerate(detections):
            if (self.yolo):
                x_min, y_min, x_max, y_max = detection.tolist()
                # print("Here 1: ", x_min,y_min,x_max,y_max)
                # x = cx - w / 2
                # y = cy - h / 2
                # x_min = int(floor(x))
                # y_min = int(floor(y))
                # x_max = int(ceil(x+w))
                # y_max = int(ceil(y+h))
            else:
                x_min = int(floor(detection.bbox.points[0].x))
                y_min = int(floor(detection.bbox.points[0].y))
                x_max = int(ceil(detection.bbox.points[2].x))
                y_max = int(ceil(detection.bbox.points[2].y))
            
            # factor to dilate the bbox
            scaleFactor = 1.2
            # width, height and center
            width = x_max - x_min
            height = y_max - y_min
            # print("Here 2: W x H :",width,height)
            center = np.array([(x_min+x_max)/2., (y_min+y_max)/2.])
            # scale of dilated image
            scalePixels = scaleFactor*max(width, height)/2
        
            # Increases the size of the patch to ensure the entire object is included
            # We also shift everything to match the preallocated image
            x_min = int(center[0] - scalePixels + im_height + im_width/2)
            x_max = int(center[0] + scalePixels + im_height + im_width/2)
            y_min = int(center[1] - scalePixels + im_width + im_height/2)
            y_max = int(center[1] + scalePixels + im_width + im_height/2)
            # print("Here 3: After increasing path size: ",x_min,y_min,x_max,y_max)
            
            # crop and resize
            # print('Here 4: image: ', image.shape)
            # print('x_min: ', x_min)
            # print('x_max: ', x_max)
            # print('y_min: ', y_min)
            # print('y_max: ', y_max)
            patch = image[y_min:y_max, x_min:x_max, :]
            # plt.figure()
            # plt.imshow(patch)
            
            resized_patch = cv2.resize(patch, (self.img_size, self.img_size))
            # plt.figure()
            # plt.title('Resized Patch')
            # plt.imshow(resized_patch)
            
            # collect patches
            resized_patch = np.moveaxis(resized_patch, 2, 0)
            # plt.figure()
            # plt.title('Move Axis Resized Patch')
            # plt.imshow(np.transpose(resized_patch, (1, 2, 0)))
            
            patches[i, :, : :] = resized_patch
        
            # Make sure to shift everything back to match the original image
            x_min = int(x_min - im_height - im_width/2)
            x_max = int(x_max - im_height - im_width/2)
            y_min = int(y_min - im_width - im_height/2)
            y_max = int(y_max - im_width - im_height/2)
        
            # print("Here 5: shifting back to center: ",x_min,y_min,x_max,y_max)
        
            bounds.append([x_min, x_max, y_min, y_max])
        
        return patches, bounds

    def get_keypoints(self, patches):
        ''' Runs the images through the network
        '''
        patches_tensor = torch.from_numpy(patches)
        patches_tensor = patches_tensor.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            heatmaps = self.model(patches_tensor)
            # print('heatmap shape:', heatmaps[-1].cpu().numpy().shape)
        # The network returns a list of the outputs of all of the hourglasses
        # in the stack.  We want the output of the final hourglass
        return heatmaps[-1].cpu().numpy()

    def generate_heatmap_grid(self, keypoints, object_type):
        ''' Generates a grid of heatmaps for a single detection
        '''
        num_images = self.keypoints_indices[object_type][1] - \
            self.keypoints_indices[object_type][0]

        grid_size = int(ceil(sqrt(num_images)))
        combined_keypoints = np.zeros((64*grid_size, 64*grid_size), dtype=np.float32)
        for i in range(grid_size):
            for j in range(grid_size):
                index = i * grid_size + j + self.keypoints_indices[object_type][0]
                if index >= self.keypoints_indices[object_type][1]:
                    continue
                # print(keypoints[index].shape)
                combined_keypoints[i*64:(i+1)*64, j*64:(j+1)*64] = keypoints[index]
        return combined_keypoints

    def track_objects(self, rgb) -> ultralytics.engine.results.Results:
        results = self.yolo.track(rgb, persist=True, classes=[2])
        return results[0]

    def detect_keypoints(self, yolo_result, image_msg):
        bounding_boxes = yolo_result.boxes.xyxy.numpy()
        print(f'Got {len(bounding_boxes)} detections in a {image_msg.shape[1]} x {image_msg.shape[0]} image')

        width = image_msg.shape[1]
        height = image_msg.shape[0]
        preallocated_img_size = [2*(width+height), 2*(width+height), 3]
        img_preallocated = np.zeros(preallocated_img_size).astype(np.float32)
        original_img_size = [height, width, 3]
        img = np.zeros(original_img_size).astype(np.float32)
        img_published = np.zeros(original_img_size).astype(np.uint8)
        

        
        img = image_msg.astype(np.float32)/255.0
        img_preallocated[width+height//2:width+height//2+height,height+width//2:height+width//2+width,:] = img

        img_published = image_msg

        bounding_boxes_detected = bounding_boxes

        
        patches, bounds = self.get_patches_and_bounds(bounding_boxes_detected, img_preallocated, height, width)


        pred_keypoints = self.get_keypoints(patches)


        detection_result = {} ## Dictionary- {Car_no : {Key_pts: __, confidence: __ }}

        
        for i, detection in enumerate(yolo_result):

            predictions = pred_keypoints[i, :, :, :]
            obj_name = yolo_result.names[int(detection.boxes.cls[0].item())]

            if(obj_name not in detection_result):
                detection_result[obj_name] = {}
                detection_result[obj_name]['2D_kpts'] = {}
                detection_result[obj_name]['prediction_confs'] = {}


            
            if obj_name not in self.keypoints_indices:
                # print('skip')
                continue

            keypts_2D_per_car = []
            confs = []
            append = True
            for j in range(self.keypoints_indices[obj_name][0], self.keypoints_indices[obj_name][1]):
                coords = np.unravel_index(np.argmax(predictions[j]), predictions[j, :, :].shape)

                # 2D Keypoints per detections is present here (in original image).
                img_coords = [0, 0]
                img_coords[0] = bounds[i][0] + int(1.0 * coords[1] / predictions.shape[-2] * (bounds[i][1] - bounds[i][0]) + 0.5)
                img_coords[1] = bounds[i][2] + int(1.0 * coords[0] / predictions.shape[-1] * (bounds[i][3] - bounds[i][2]) + 0.5)
                
                conf_per_kp = predictions[j, coords[0], coords[1]]
                confs.append(conf_per_kp)
                # Red if the keypoint is less than our threshold, green otherwise. This is Keypoints_2d,
                if conf_per_kp < self.activation_threshold:
                    append=False
                    color = (255, 0, 0)
                    print(f'skipping {i}th car ....')
                    break
                else:
                    color = (0, 255, 0)
                # print('img_coords: ', img_coords)    
                cv2.circle(img_published, (img_coords[0], img_coords[1]), 5, color, thickness=-1)

                # Annotate the circle with text
                text = str(j)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                text_color = (255, 0, 0)  # Black text
                text_thickness = 2
                
                # Adjust text position slightly above the circle
                text_position = (img_coords[0] + 10, img_coords[1] - 10)  # Offset to avoid overlap
                
                cv2.putText(img_published, text, text_position, font, font_scale, text_color, text_thickness)

                keypts_2D_per_car.append(img_coords)
            if append:
                detection_result[obj_name]['2D_kpts'][i] = torch.Tensor(keypts_2D_per_car)
                detection_result[obj_name]['prediction_confs'][i] = torch.Tensor(confs)

            else:
                detection_result[obj_name]['2D_kpts'][i] = None
                detection_result[obj_name]['prediction_confs'][i] = None


        # plt.figure(figsize=(10, 8))
        # plt.imshow(img_published)
        # plt.savefig("./output/kpt_image.png", dpi=1200)
        return pred_keypoints, patches, detection_result

    def heatmaps_get_conf(self, heatmaps):
        # heatmaps = heatmaps.numpy()
        conf = np.max(heatmaps, axis=(-2,-1))
        return conf


    def mapped_confidence(self, confs):
        mapped_conf = np.zeros(12)
        mapped_conf[0] = confs[8]
        mapped_conf[1] = confs[9]
        mapped_conf[2] = confs[10]
        mapped_conf[3] = confs[11]
        
        mapped_conf[4] = confs[0]
        mapped_conf[5] = confs[1]
        mapped_conf[6] = confs[2]
        mapped_conf[7] = confs[3]
        
        mapped_conf[8] = confs[4]
        mapped_conf[9] = confs[5]
        mapped_conf[10] = confs[6]
        mapped_conf[11] = confs[7]

        return torch.from_numpy(mapped_conf)

    # Weak Perspective 

    def WeakPerspective(self, r, t_est, kpts_3D, norm_2D_Kpts, d, log=False):
        # print("Weak Perspective ...")
        optimizer = torch.optim.Adam([r,t_est], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.7)
        prev_loss = float('inf')
        err_plot = []
            
        while True:
            optimizer.zero_grad()
            R_est = rodrigues(r)
            S_cam_regular = (torch.matmul(R_est,kpts_3D.transpose(1,0))+t_est[:,None])
            
            S_cam_optical = torch.matmul(self.R_reg_2_opt,S_cam_regular)
            
            err = (norm_2D_Kpts[0:2,:] - S_cam_optical[0:2,:])*d.sqrt()
            
            err_val = (torch.norm(err,'fro')**2)/2
            
            err_val.backward()
            optimizer.step()
            scheduler.step()

            if log:
                print(err_val.item())
                err_plot.append(err_val.item())
            
            if abs(prev_loss-err_val.item())/prev_loss<1e-7:
                break
            
            prev_loss = err_val.item()

        return r, t_est


    # Full Perspective
    def FullPerspective(self, r, t_est, kpts_3D, norm_2D_Kpts, d, log=False):
        # print("Full Perspective ...")
        Z = torch.rand(12, requires_grad=True, device='cpu')
        optimizer_1 = torch.optim.Adam([Z,r,t_est], lr=1e-3)
        prev_loss = float('inf')    
        err_plot = []
        
        while True:
            optimizer_1.zero_grad()
            
            R_est = rodrigues(r)
            
            S_cam_frame = torch.matmul(R_est,kpts_3D.transpose(1,0))+t_est[:,None]
        
            S_optical_frame = torch.matmul(self.R_reg_2_opt,S_cam_frame)
            
            err = (S_optical_frame - (norm_2D_Kpts*Z))*d.sqrt()
            
            err_val = (torch.norm(err,'fro')**2)/2
            
            # Backpropagation
            err_val.backward()
            optimizer_1.step()

            if log:
                print( err_val.item())
                err_plot.append(err_val.item())
        
            if abs(prev_loss-err_val.item())/prev_loss<1e-10:
                break
        
            
            prev_loss = err_val.item()

        return r, t_est
    
    def GetCarPoses(self, img, yolo_result):
        print("Estimating Car pose ...")
        _, _, detection_result = self.detect_keypoints(yolo_result,img)
        
        plot_bboxes(img,yolo_result.boxes.xywh)

        result_dict = {}

        kpts_3D = S.to('cpu')
        for car_no in detection_result['car']['2D_kpts']:

            if(detection_result['car']['2D_kpts'][car_no] == None):
                print(f'Skipping car no: {car_no} because of low confidence')
                continue

            d = self.mapped_confidence(detection_result['car']['prediction_confs'][car_no]).to('cpu')
            kpts_2D = detection_result['car']['2D_kpts'][car_no].float().to('cpu')
            norm_2D_Kpts = torch.matmul(self.K.inverse(),torch.vstack((kpts_2D.transpose(1,0),torch.ones(1,12))))
            
            # Initialize random r and t
            r = torch.rand(3, requires_grad=True, device='cpu') # rotation in axis-angle representation
            t_est = torch.rand(3 ,requires_grad=True, device='cpu')

            r_wp, t_wp = self.WeakPerspective(r,t_est, kpts_3D.to('cpu') , norm_2D_Kpts, d)
            # print("r : ", r_wp)
            # print("t : ", t_wp)

            r_final, t_final = self.FullPerspective(r_wp, t_wp, kpts_3D, norm_2D_Kpts, d)
            # print(f'Pose of car no: {car_no} is :')
            # print("r (axis angle): ", r_final.detach())
            # print("t : ", t_final.detach())

            car_pose = {'r':r_final.detach(), 't': t_final.detach()}
            result_dict[car_no] = car_pose

        return result_dict



if __name__ == '__main__':
        
    img_path = './output/000117.png'
    PoseEstObj = PoseEstimation()

    yolo_result = PoseEstObj.track_objects(img_path)

    img = cv2.imread(img_path) # Load the image


    # # ========================================================

    # plot_bboxes(img_path,yolo_result.boxes.xywh) ##  Plot BBox on the image, for sanity check
    # Get all Car related prediction points 
    # pred_keypts_for_car = pred_keypoints[:,PoseEstObj.keypoints_indices['car'][0]:PoseEstObj.keypoints_indices['car'][1],:,:]

    # pred_keypoints, car_patches, detection_result = PoseEstObj.detect_keypoints(yolo_result,img)
    # car_no = 1
    # kpts_3D = S.to('cpu')
    # for car_no in detection_result['car']['2D_kpts']:

    #     if(detection_result['car']['2D_kpts'][car_no] == None):
    #         print(f'Skipping car no: {car_no} because of low confidence')
    #         continue

    #     d = PoseEstObj.mapped_confidence(detection_result['car']['prediction_confs'][car_no]).to('cpu')
    #     kpts_2D = detection_result['car']['2D_kpts'][car_no].float().to('cpu')
    #     norm_2D_Kpts = torch.matmul(PoseEstObj.K.inverse(),torch.vstack((kpts_2D.transpose(1,0),torch.ones(1,12))))
        
    #     # Initialize random r and t
    #     r = torch.rand(3, requires_grad=True, device='cpu') # rotation in axis-angle representation
    #     t_est = torch.rand(3 ,requires_grad=True, device='cpu')

    #     r_wp, t_wp = PoseEstObj.WeakPerspective(r,t_est, kpts_3D.to('cpu') , norm_2D_Kpts, d)
    #     # print("r : ", r_wp)
    #     # print("t : ", t_wp)

    #     r_final, t_final = PoseEstObj.FullPerspective(r_wp, t_wp, kpts_3D, norm_2D_Kpts, d)
    #     print(f'Pose of car no: {car_no} is :')
    #     print("r (axis angle): ", r_final)
    #     print("t : ", t_final)

    # # ============================================================

    car_poses = PoseEstObj.GetCarPoses(img, yolo_result)
    print("Final Car Poses: ")
    print(car_poses)




    # d = PoseEstObj.mapped_confidence(car_pred_keypts_conf_score[car_no]).to('cpu')
    # kpts_2D = detection_result[car_no,:,:].float().to('cpu')
    # kpts_3D = S.to('cpu')
    # # K = K.to('cpu')

    # norm_2D_Kpts = torch.matmul(PoseEstObj.K.inverse(),torch.vstack((kpts_2D.transpose(1,0),torch.ones(1,12))))
    # # norm_2D_Kpts

    # # Initialize random r and t
    # r = torch.rand(3, requires_grad=True, device='cpu') #torch.rand(3, requires_grad=True, device='cpu') # rotation in axis-angle representation
    # t_est = torch.rand(3 ,requires_grad=True, device='cpu')

    # r_wp, t_wp = PoseEstObj.WeakPerspective(r,t_est, kpts_3D.to('cpu') , norm_2D_Kpts, d)
    # print("r : ", r_wp)
    # print("t : ", t_wp)

    # r_final, t_final = PoseEstObj.FullPerspective(r_wp, t_wp, kpts_3D, norm_2D_Kpts, d)
    # print("r : ", r_final)
    # print("t : ", t_final)