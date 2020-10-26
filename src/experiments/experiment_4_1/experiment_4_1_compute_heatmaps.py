import os
import pickle
import numpy as np
from tqdm import tqdm
from time import sleep

from utils.viz import save_image, get_image, get_normalized_image, normalize, get_heatmap
from utils.prepare_models import get_model
from model.gradcam import GradCAM

ground_truth_resized_bounding_boxes_path = './ground_truth_bounding_boxes_original.pickle'
VOC2007_images_directory = '../../../datasets/VOC/VOC2007/JPEGImages/'


if __name__ == "__main__":
    # please see prepare_models.py for a list of valid models
    # Dated: 21.10.2020; valid_models = ['resnet50','vgg16','googlenet','inception_v3', 'alexnet']
    model_name = 'resnet50'
    
    # Retrieve the model
    model, layer_name = get_model(model_name)

    # Instantiate the gradCAM class
    gradcam = GradCAM(  model=model, \
                        target_layer = layer_name
                        )

    # Read the ground truth file to get list of images to process
    with open(ground_truth_resized_bounding_boxes_path,'rb') as opened_file:
        ground_truth_bounding_boxes = pickle.load(opened_file)

    # Iterate through every image and compute bounding boxes
    for image_identifier in tqdm(list(ground_truth_bounding_boxes.keys())):
        
        # Strip the .xml from the identifier
        image_identifier = image_identifier.strip('.xml')

        # Define the image path
        image_path = VOC2007_images_directory + image_identifier + '.jpg'

        # Obtain normalized image
        normalized_image = get_normalized_image(image_path)

        # Obtain the image mask by applying gradcam
        mask = gradcam(normalized_image)

        # Obtain the result of merging the mask with the torch image
        heatmap = get_heatmap(mask)

        # Save the heatmap!
        save_image(heatmap, f'../../../results/experiment_4_1/heatmaps/{image_identifier}.png')


                
