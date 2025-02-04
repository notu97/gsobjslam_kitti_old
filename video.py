import cv2
import os
from natsort import natsorted

def create_video_from_images(image_folder, output_video, fps):
    # Get list of all files in the directory
    files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # Sort files naturally (human order)
    files = natsorted(files)

    # Read the first image to get the width and height
    first_image_path = os.path.join(image_folder, files[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Read each file and write to the video
    for file in files:
        image_path = os.path.join(image_folder, file)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the VideoWriter
    video.release()

# Usage
image_folder = 'output/KITTI/0/mapping_vis'
output_video = 'output/KITTI/0/output_video.mp4'
fps = 10

create_video_from_images(image_folder, output_video, fps)