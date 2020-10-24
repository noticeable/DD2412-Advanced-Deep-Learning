import os
import pickle
import numpy as np
from tqdm import tqdm
from time import sleep

from utils.viz import save_image, get_image, get_normalized_image, normalize, get_heatmap
from utils.prepare_models import get_model
from model.gradcam import GradCAM

DEBUG = False

"""
#   Computes the gradCAM heatmap for a given image
#   :param  gradCAM     gradCAM object to call for computing results
#   :param  image_path  directory location of image
#   :output heatmap     heatmap generated from GradCAM processing
"""
def compute_heatmap(gradcam, image_path):

    # Obtain normalized image
    normalized_image = get_normalized_image(image_path)

    # Obtain the image mask by applying gradcam
    mask = gradcam(normalized_image)

    # Obtain the result of merging the mask with the torch image
    heatmap = get_heatmap(mask)

    return heatmap


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

    # Mapping of input_list key to file name and vice versa
    input_key_map = {}
    key_input_map = {}

    # Directory to the ground truth localization cues to be modified
    localization_cues_file          = './sample_localization_cues/localization_cues.pickle'

    # Directory to list of training samples
    input_list_path                 = './SEC_pytorch/datalist/PascalVOC/input_list.txt'
    VOC_root_path                   = '../../../datasets/VOC/VOC2012/'
    VOC_input_image_directory       = VOC_root_path + 'JPEGImages/'


    # Populate the input list identifier to key mapping and vice versa
    input_list = open(input_list_path, 'r')
    for entry in input_list:
        file_identifier, key = entry.split()
        input_key_map[file_identifier] = key
        key_input_map[key] = file_identifier


    # Open the existing localization_cues with ground truth provided
    # Dictionary which stores two items per image identifier 
    # {image_identifier}_labels : [class1, class2...,classn]
    # {image_identifier}_cues: array([[matrix],[matrix],[matrix]]) = array(3D-Matrix)
    file = open(localization_cues_file,'rb')
    data = pickle.load(file)

    for line in tqdm(data):

        # Update the tqdm status bar
        sleep(0.25)

        # If heatmaps are provided, regnerate new heat maps using gradCAM
        if '_cues' in line:

            # Find corresponding image identifier
            image_identifier = key_input_map[line.strip('_cues')]
            
            # Specify the directory to retrieve the test identifier's image
            image_path = VOC_input_image_directory + image_identifier

            # Specity the directory to retrieve the test identifier's annotations
            heatmap = compute_heatmap(gradcam, image_path)
            
            # Store the new heatmap into the pickle dataset
            data[line] = heatmap

    # Save the updated pickle file!
    pickle.dump('localization_cues_BY.pickle', data)

    assert(False)
