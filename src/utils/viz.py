# contains helper functions for visualising grad cam/guided grad cam/etc on images
import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from skimage import io

"""
# Retreives and prepare the image for processing [ not normalized]
# inputs:
#   path: string: path to the image
# outputs:
#   image: tensor: tensor representing the image under investigation
"""
def get_image(path='../../datasets/test_images/cat_dog.png'):
    image = io.imread(path)
    image = np.float32(cv2.resize(image, (224, 224))) / 255

    return image


"""
# Retreives and prepare a normalized image for processing
# inputs:
#   path: string: path to the image
# outputs:
#   input_image: tensor: tensor representing the image under investigation
"""
def get_normalized_image(path='../../datasets/test_images/cat_dog.png'):
    image = io.imread(path)
    image = np.float32(cv2.resize(image, (224, 224))) / 255

    # Convert the image into torch format
    input_image = image.copy()
    input_image -= np.array([0.485, 0.456, 0.406])
    input_image /= np.array([0.229, 0.224, 0.225])
    input_image = np.ascontiguousarray(np.transpose(input_image, (2, 0, 1)))
    input_image = input_image[np.newaxis, ...]

    input_image = torch.tensor(input_image, requires_grad=True)
    return input_image


"""
# Saves the image
# inputs:
#   image:  tensor: tensor of image to save
#   path:   string: path of where to save the image

"""
def save_image(image, path='../../results/test_results/cat_dog_guided_backpropagation.png'):
    io.imsave(path, image)


"""
# Normalize the image to the [0,255] range
# inputs:
#   input_image:    np.matrix: image to be normalized
# outputs:
#   image:          np.matrix: normalized image
"""
def normalize(input_image):
    image = input_image.copy()
    image = image - np.max(np.min(image),0)
    image /= np.max(image) 
    image *= 255
    image = np.uint8(image)

    return image


"""
# Produces the heatmap from gradCAM results
# inputs:
#   mask:   np.matrix: mask output from gradCAM forward pass
# outputs;
#   heatmap: np.matrix: color adjusted heatmap
"""
def get_heatmap(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)/255
    heatmap = heatmap[...,::-1]

    return heatmap