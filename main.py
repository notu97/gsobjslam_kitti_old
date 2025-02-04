from utils.utils import *
from datasets import *
from GS_OBJ_SLAM import *


configs_path = 'configs/kitti.yaml'
slam = GS_OBJ_SLAM(configs_path)
slam.run()

print(slam.associations)
