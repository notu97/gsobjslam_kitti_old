# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

matplotlib.use('TkAgg')


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


# 3D Plot of points
def plot_3D_pts(array):
    
    # front: x, left: y, upwards: z, array = torch.matmul(R_90,S.transpose(1,0)).transpose(1,0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    for i in range(12): #plot each point + it's index as text above
        ax.scatter(array[i,0], array[i,1], array[i,2], color='b') 
        ax.text(array[i,0], array[i,1], array[i,2],  '%s' % (str(i)), size=20, zorder=1,color='k')

    ax.scatter(0,0,0,color='red')
    ax.quiver(0,0,0,0.5,0,0,color='r')
    ax.quiver(0,0,0,0,0.5,0,color='g')
    ax.quiver(0,0,0,0,0,0.5,color='b')
    ax.set_aspect('equal')


    # Labels
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    # ax.set_title("3D Scatter Plot")
    ax.set_title("3D Scatter Plot with Rotation Axes")

    plt.show()

def plot_bboxes(image_path, bboxes):
    """
    Plots bounding boxes on an RGB image.
    
    Parameters:
        image_path (str): Path to the RGB image.
        bboxes (torch.Tensor): Tensor of shape (N, 4) with bounding boxes in (x, y, w, h) format.
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Plot each bounding box
    i=0
    for bbox in bboxes:
        cx, cy, w, h = bbox.tolist()
        x = cx - w / 2
        y = cy - h / 2
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y,  '%s' % (str(i)), size=20, zorder=1,color='r')
        i = i+1
    
    # Show the image with bounding boxes
    plt.axis('off')
    plt.show()