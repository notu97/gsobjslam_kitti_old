import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import pykitti
from third_party.semantic_kitti_api.auxiliary.laserscan import SemLaserScan
import yaml


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_config: dict):
        if dataset_config["input_path"] is not None:
            self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)


class KITTI(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        # find data
        self.kitti_root = dataset_config["kitti_root"]
        self.raw_data_root = dataset_config["raw_data_root"]
        self.semkitti_root = dataset_config["semkitti_root"]
        self.semkitti_config = dataset_config["semkitti_config"]
        self.seq_id = dataset_config["seq_id"]
        self.mapping = {
            "00": "2011_10_03_drive_0027",
            "01": "2011_10_03_drive_0042",
            "02": "2011_10_03_drive_0034",
            "03": "2011_09_26_drive_0067",
            "04": "2011_09_30_drive_0016",
            "05": "2011_09_30_drive_0018",
            "06": "2011_09_30_drive_0020",
            "07": "2011_09_30_drive_0027",
            "08": "2011_09_30_drive_0028",
            "09": "2011_09_30_drive_0033",
            "10": "2011_09_30_drive_0034"
        }
        self.seq_lenths = {0: 4541, 5: 2761, 6: 1101, 8: 4071}
        self.drive_dict = {0: "0042", 5: "0020", 6: "0020", 8: "0020"}
        self.seq_len = self.seq_lenths[self.seq_id]
        self.date_drive = self.mapping["%02d" % self.seq_id]
        self.date = self.date_drive[:10]
        self.drive = self.drive_dict[self.seq_id]
        # get o2Tv
        raw_data = pykitti.raw(self.raw_data_root, self.date, self.drive)
        self.o2Tv = raw_data.calib.T_cam2_velo
        # get intrinsics
        P2 = raw_data.calib.P_rect_20
        self.intrinsics = P2[:3, :3]
        # semkitti
        self.semkitti_config_file = yaml.safe_load(open(self.semkitti_config, 'r'))
        self.color_dict = self.semkitti_config_file['color_map']
        self.nclasses = len(self.color_dict)
        # poses
        pose_list = os.path.join(self.semkitti_root, '%02d' % self.seq_id, 'poses.txt')
        self.pose_data = self.parse_list(pose_list, skiprows=0)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        return np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)

    def __getitem__(self, index):
        frame_id = index
        # load image
        left_img_file = os.path.join(self.kitti_root, 'sequences/%02d' % self.seq_id, 'image_2', '%06d.png' % frame_id)
        left_img = cv2.imread(str(left_img_file))
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        # load lidar
        lidar_file = os.path.join(self.semkitti_root, '%02d' % self.seq_id, 'velodyne', '%06d.bin' % frame_id)
        scan = SemLaserScan(self.nclasses, self.color_dict, project=False)
        scan.open_scan(lidar_file)
        lidar_all = scan.points
        # load pose
        pose_vec = self.pose_data[frame_id, :].astype(np.float64)
        pose_vec = np.concatenate([pose_vec, np.array([0, 0, 0, 1])])
        c2w = pose_vec.reshape(4, 4)

        return frame_id, left_img, lidar_all, c2w.astype(np.float32)

    def __len__(self):
        return self.seq_len if self.frame_limit < 0 else int(self.frame_limit)


def get_dataset(dataset_name: str):

    if dataset_name == "kitti":
        return KITTI
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
