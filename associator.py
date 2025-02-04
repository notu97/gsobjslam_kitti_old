import ultralytics.engine.results
from utils.associator_utils import *
from utils.utils import get_render_settings


class Associator:

    def __init__(self, configs: dict) -> None:

        self.configs = configs
        self.iou_thresh = self.configs['iou_thresh']

    def associate(self, yolo_result: ultralytics.engine.results.Results, submaps: list,
                  c2w: np.ndarray, intrinsics: np.ndarray) -> dict:

        # yolo_bboxes = bboxes_from_tracker(yolo_result, gt_color, gt_depth, c2w, intrinsics)
        # model_bboxes = bboxes_from_gaussians([m[1] for m in submaps])
        # iou_matrix = iou_3d_batch(yolo_bboxes, model_bboxes)

        yolo_segs = seg_from_tracker(yolo_result, H=self.configs['H'], W=self.configs['W'])
        w2c = np.linalg.inv(c2w)
        render_settings = get_render_settings(self.configs['W'], self.configs['H'], intrinsics, w2c)
        model_segs = seg_from_gaussians([m[1] for m in submaps], render_settings)
        iou_matrix = iou_seg_batch(yolo_segs, model_segs)

        cost = iou_matrix
        cost[np.where(cost < self.iou_thresh)] = -1e6
        cost = -cost
        # shape (N, 2), each row [idx_of_yolo_bbox, idx_of_submap_bbox]
        matched_indices = linear_assignment(cost)

        new_associations = {}
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] > self.iou_thresh:
                tracker_id = int(yolo_result.boxes.id[m[0]].numpy())
                new_associations[tracker_id] = submaps[m[1]][0]

        return new_associations
