import numpy as np
import ultralytics.engine.results
from argparse import ArgumentParser
import torchvision
import time
from matplotlib import pyplot as plt

from gaussian_model import GaussianModel
from arguments import OptimizationParams
from datasets import *
from utils.utils import *
from utils.mapper_utils import *
from losses import *


class Mapper:

    def __init__(self, config: dict, dataset: BaseDataset) -> None:

        self.config = config
        self.alpha_thre = config['alpha_thre']
        self.uniform_seed_interval = config['uniform_seed_interval']
        self.iterations = config['iterations']
        self.new_submap_iterations = config['new_submap_iterations']
        self.pruning_thre = config['pruning_thre']

        self.dataset = dataset
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))

    def new(self, frame_id: int, c2w: np.ndarray, yolo_result: ultralytics.engine.results.Results,
            object_idx: int) -> GaussianModel:

        gs = GaussianModel(0)
        gs.training_setup(self.opt)
        gs = self.update(frame_id, c2w, yolo_result, gs, object_idx, is_new=True)

        return gs

    def update(self, frame_id: int, c2w: np.ndarray, yolo_result: ultralytics.engine.results.Results,
               submap: GaussianModel, object_idx: int, is_new=False) -> GaussianModel:

        _, gt_color, lidar, _ = self.dataset[frame_id]
        w2c = np.linalg.inv(c2w)
        obj_mask = torch.squeeze(yolo_result.masks[object_idx].data).cpu().numpy()
        obj_mask = cv2.resize(obj_mask, (self.dataset.width, self.dataset.height),
                              interpolation=cv2.INTER_NEAREST)
        keyframe = {
            "img": gt_color,
            "lidar": lidar,
            "mask": obj_mask,
            "c2w": c2w,
            "render_settings": get_render_settings(
                self.dataset.width, self.dataset.height, self.dataset.intrinsics, w2c)}

        seeding_mask = self.compute_seeding_mask(submap, keyframe, is_new)
        pts, uv = pcd_with_mask(keyframe['lidar'], keyframe['img'], seeding_mask, self.dataset.o2Tv,
                            self.dataset.intrinsics, keyframe["c2w"])
        new_pts_num = self.grow_submap(c2w, submap, pts)
        # print("New points num: %d" % new_pts_num)

        # create depth from lidar points
        depth = np.zeros((self.dataset.height, self.dataset.width), dtype=np.float32)
        depth[uv[1, :].astype(int), uv[0, :].astype(int)] = pts[:, 2]
        depth = depth * obj_mask.astype(float)
        keyframe['depth'] = depth
        # optimize
        max_iterations = self.iterations
        if is_new:
            max_iterations = self.new_submap_iterations
        opt_dict = self.optimize_submap([(frame_id, keyframe)], submap, max_iterations)
        optimization_time = opt_dict['optimization_time']
        print("Optimization time: ", optimization_time)

        return submap

    def compute_seeding_mask(self, gaussian_model: GaussianModel, keyframe: dict, is_new: bool) -> np.ndarray:

        if is_new:
            seeding_mask = keyframe['mask'].astype(bool)
        else:
            render_dict = render_gs([gaussian_model], keyframe["render_settings"])
            alpha_mask = (render_dict["alpha"] < self.alpha_thre)
            seeding_mask = torch2np(alpha_mask[0, :, :]) & keyframe['mask'].astype(bool)

        return seeding_mask

    def grow_submap(self, c2w: np.ndarray, submap: GaussianModel, pts: np.ndarray) -> int:

        # @TODO: filter the points
        new_pts_ids = np.arange(pts.shape[0])
        cloud_to_add = np2ptcloud(pts[new_pts_ids, :3], pts[new_pts_ids, 3:] / 255.0)
        submap.add_points(cloud_to_add)
        submap._features_dc.requires_grad = False
        submap._features_rest.requires_grad = False
        # @TODO: Re-enable terminal output
        print("Gaussian model size", submap.get_size())

        return new_pts_ids.shape[0]

    def optimize_submap(self, keyframes: list, submap: GaussianModel, iterations: int = 100) -> dict:

        iteration = 0
        losses_dict = {}

        start_time = time.time()
        while iteration < iterations + 1:
            submap.optimizer.zero_grad(set_to_none=True)
            # @TODO: optimize using multiple views
            keyframe_id = 0

            frame_id, keyframe = keyframes[keyframe_id]
            render_pkg = render_gs([submap], keyframe["render_settings"])

            image, depth = render_pkg["color"], render_pkg["depth"]
            color_transform = torchvision.transforms.ToTensor()
            gt_image = color_transform(keyframe["img"]).cuda()
            gt_depth = np2torch(keyframe["depth"], device='cuda')

            depth_mask = (gt_depth > 0) & (~torch.isnan(depth)).squeeze(0)
            obj_mask = torch.from_numpy(keyframe["mask"]).bool().cuda()
            depth_mask = ~obj_mask | depth_mask
            # directly use gt_image because it is already masked
            color_loss = (1.0 - self.opt.lambda_dssim) * l1_loss(image, gt_image) + \
                         self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            depth_loss = l1_loss(depth[:, depth_mask], gt_depth[depth_mask])
            reg_loss = isotropic_loss(submap.get_scaling())
            total_loss = color_loss + depth_loss + reg_loss
            total_loss.backward()

            losses_dict[frame_id] = {"color_loss": color_loss.item(),
                                     "depth_loss": depth_loss.item(),
                                     "total_loss": total_loss.item()}

            with torch.no_grad():

                if iteration == iterations // 2 or iteration == iterations:
                    prune_mask = (submap.get_opacity()
                                  < self.pruning_thre).squeeze()
                    submap.prune_points(prune_mask)

                # Optimizer step
                if iteration < iterations:
                    submap.optimizer.step()
                submap.optimizer.zero_grad(set_to_none=True)

            iteration += 1
        optimization_time = time.time() - start_time
        losses_dict["optimization_time"] = optimization_time
        losses_dict["optimization_iter_time"] = optimization_time / iterations
        return losses_dict
