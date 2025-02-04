import numpy as np
import ultralytics.engine.results
from ultralytics import YOLO
import torchvision
import cv2

from utils.associator_utils import seg_from_gaussians
from utils.utils import *
from datasets import *
from associator import Associator
from mapper import Mapper
from logger import Logger


class GS_OBJ_SLAM(object):

    def __init__(self, configs_path: str) -> None:

        self.configs = load_config(configs_path)
        self.dataset = get_dataset(self.configs['dataset_name'])(self.configs)
        self.yolo = YOLO("yolo_models/yolo11x-seg.pt")  # load an official model
        self.associator = Associator(self.configs)
        self.mapper = Mapper(self.configs, self.dataset)
        self.logger = Logger(self.configs['output_path'])

        self.gaussian_models = []   # each object as a submap
        self.associations = {}  # key: tracking id from YOLO. value: idx of submap.

        self.cur_gt_color = None
        self.cur_lidar = None
        self.cur_est_c2w = None

    def track_objects(self, rgb) -> ultralytics.engine.results.Results:

        results = self.yolo.track(rgb, persist=True, classes=[2])

        return results[0]

    def update_associations(self, yolo_result: ultralytics.engine.results.Results) -> dict:

        ids = list(np.int32(yolo_result.boxes.id.numpy()))
        new_detections = yolo_result[[i for i, id in enumerate(ids) if id not in self.associations.keys()]]
        old_ids = [id for id in ids if id in self.associations.keys()]
        associated_models_idxs = [self.associations[id] for id in old_ids]
        dangling_models = [(i, model) for i, model in enumerate(self.gaussian_models)
                           if i not in associated_models_idxs]

        if (len(dangling_models) == 0) or (len(new_detections) == 0):
            return {}

        new_associations = self.associator.associate(new_detections, dangling_models,
                                                     self.cur_est_c2w, self.dataset.intrinsics)
        self.associations.update(new_associations)
        # print("New associations found:")
        # print(new_associations)

        return new_associations

    def run(self) -> None:

        for frame_id in range(len(self.dataset)):

            _, self.cur_gt_color, self.cur_lidar, gt_pose = self.dataset[frame_id]
            if self.configs['gt_camera']:
                self.cur_est_c2w = gt_pose
            else:
                raise NotImplementedError

            # track objects
            yolo_result = self.track_objects(self.cur_gt_color)
            if yolo_result.boxes.id is None:
                continue
            # update associations
            self.update_associations(yolo_result)

            # iterate over objects
            for i in range(len(yolo_result)):
                tracking_id = int(yolo_result.boxes.id[i].numpy())
                # if this tracking id is already associated
                if tracking_id in self.associations.keys():
                    # optimize associated submap
                    self.mapper.update(frame_id, self.cur_est_c2w, yolo_result,
                                       self.gaussian_models[self.associations[tracking_id]], i)
                else:
                    print('New tracking id: %s' % tracking_id)
                    # start a new submap
                    self.gaussian_models.append(self.mapper.new(frame_id, self.cur_est_c2w, yolo_result, i))
                    self.associations[tracking_id] = len(self.gaussian_models) - 1  # record association

            # Visualise the mapping for the current frame
            w2c = np.linalg.inv(self.cur_est_c2w)
            color_transform = torchvision.transforms.ToTensor()
            keyframe = {
                "color": color_transform(self.cur_gt_color).cuda(),
                "depth": np2torch(np.zeros_like(self.cur_gt_color[:, :, 0]), device="cuda"),
                "render_settings": get_render_settings(
                    self.dataset.width, self.dataset.height, self.dataset.intrinsics, w2c)}

            depths = seg_from_gaussians(self.gaussian_models, keyframe['render_settings'])

            with torch.no_grad():
                render_pkg_vis = render_gs(self.gaussian_models, keyframe['render_settings'])
                image_vis, depth_vis = render_pkg_vis["color"], render_pkg_vis["depth"]
                image_vis = cv2.cvtColor(torch2np(image_vis.permute(1, 2, 0) * 255), cv2.COLOR_BGR2RGB)
                cv2.imwrite('output/KITTI/0/mapping_vis/{:03d}.jpg'.format(frame_id), image_vis)

                # self.logger.vis_mapping_iteration(
                #     frame_id, 0,
                #     image_vis.clone().detach().permute(1, 2, 0),
                #     depth_vis.clone().detach().permute(1, 2, 0),
                #     keyframe["color"].permute(1, 2, 0),
                #     keyframe["depth"].unsqueeze(-1),
                #     yolo_result)
