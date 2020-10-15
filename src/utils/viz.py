# contains helper functions for visualising grad cam/guided grad cam/etc on images
import cv2
import numpy as np
import os

def viz_gradcam(img, mask, output_file):
    # applies mask to image to show grad-cam/etc result
    if not os.path.isdir(os.path.dirname(output_file)):
        raise NotADirectoryError
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_viz = heatmap + np.float32(img)
    cam_viz = cam_viz / np.max(cam_viz)
    cv2.imwrite(output_file, np.uint8(255 * cam_viz))
