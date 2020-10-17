# contains helper functions for visualising grad cam/guided grad cam/etc on images
import cv2
import numpy as np
import os
import torch
import PIL
from torchvision import transforms


def viz_gradcam(img, mask, output_file):
    # applies mask to image to show grad-cam/etc result
    if not os.path.isdir(os.path.dirname(output_file)):
        raise NotADirectoryError

    # Convert from torch to numpy matrix
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()

    # Apply color scheme as per the heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Move the dimension from [224,224,3] to [3,224,224]
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

    # Extract each color channel
    b, g, r = heatmap.split(1)

    # Recreate into standard rgb representation
    heatmap = torch.cat([r, g, b])

    # Merge the heatmap with the original image
    result = heatmap + img

    # Normalization
    result = result.div(result.max()).squeeze()

    # Convert the image back into PIL format
    result = transforms.ToPILImage()(result)
    
    # save
    result.save(output_file)
