import numpy as np
import torch
import torch.nn.functional as F
from utils.pose_utils import compute_camera_opt_params, multiply_quaternions
from gaussian_model import GaussianModel
from utils.gaussian_model_utils import build_rotation
from utils.utils import render_gs


def pcd_with_mask(lidar_all: np.ndarray, img: np.ndarray, mask: np.ndarray,
                  o2Tv: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> tuple:
    """
    Create colored point cloud from raw lidar scan, rgb image and mask
    Args:
        lidar_all: raw lidar scan, (N, 3)
        img: rgb image, (H, W, 3)
        mask: mask, (H, W)
        o2Tv: camera to lidar transformation, (4, 4)
        intrinsics: camera intrinsics, (3, 3)
    Returns: masked and colored point cloud, (N', 6)
             pixel coordinates of the points, (2, N')
    """
    # select lidar points that fall in camera
    lidar_all_incam = o2Tv[:3, :3] @ lidar_all.T + o2Tv[:3, 3][:, None]
    valid_mask_incam = lidar_all_incam[2, :] > 0
    lidar_pnts = lidar_all_incam[:, valid_mask_incam].T
    # convert lidar points to uvd
    normalized_pnts = lidar_pnts / lidar_pnts[:, 2][:, None]
    uv = (intrinsics @ normalized_pnts.T)[:2, :].astype(int)
    valid_uv_mask = (uv[0, :] > 0) & (uv[0, :] < img.shape[1]) \
                    & (uv[1, :] > 0) & (uv[1, :] < img.shape[0])
    uv = uv[:, valid_uv_mask]
    lidar_pnts = lidar_pnts[valid_uv_mask, :]
    inside_uv_mask = mask[uv[1, :], uv[0, :]].astype(bool)
    uv = uv[:, inside_uv_mask]
    d = lidar_pnts[inside_uv_mask, 2]
    # create the point cloud
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    xx = uv[0, :] - cx
    yy = uv[1, :] - cy
    xx = xx * d / fx
    yy = yy * d / fy
    zz = d
    pcd = np.stack([xx, yy, zz, np.ones_like(zz)], axis=1)
    # Transform points to world coordinates
    posed_points = pose @ pcd.T
    posed_points = posed_points.T[:, :3]
    # add color
    rgb = img[uv[1, :], uv[0, :], :]
    pcd = np.concatenate([posed_points, rgb], axis=1)

    return pcd, uv
