import os
import pickle
import numpy as np
from tqdm import tqdm
from time import sleep

from utils.viz import save_image, get_image, get_normalized_image, normalize, get_heatmap
from utils.prepare_models import get_model
from model.gradcam import GradCAM

DEBUG = False
NUMBER_OF_CLASSES = 20
LOCALIZATION_MAP_SIZE = 41
MODEL_WEIGHTS = '../../../weights/resnet50_finetrained.ckpt'


"""
#   Computes the gradCAM heatmap for a given image
#   :param  gradCAM     gradCAM object to call for computing results
#   :param  image_path  directory location of image
#   :output heatmap     heatmap generated from GradCAM processing
"""
def compute_heatmap(gradcam, image_path, target_class):

    # Obtain normalized image
    normalized_image = get_normalized_image(image_path)

    # Obtain the image mask by applying gradcam
    mask = gradcam(normalized_image)

    # Obtain the result of merging the mask with the torch image
    heatmap = get_heatmap(mask, target_index=target_class)

    return heatmap


if __name__ == "__main__":

    # please see prepare_models.py for a list of valid models
    # Dated: 21.10.2020; valid_models = ['resnet50','vgg16','googlenet','inception_v3', 'alexnet']
    model_name = 'resnet50'
    
    # Retrieve the model
    print(f'Loading weights for model: { model_name}...')
    model, layer_name = load_fine_trained(model_name, model_weights)
    print(f'Weights loaded!')

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

        # If heatmaps are provided, regnerate new heat maps using gradCAM
        if '_cues' in line:

            # Find corresponding image identifier
            image_identifier = key_input_map[line.strip('_cues')]
            
            # Specify the directory to retrieve the test identifier's image
            image_path = VOC_input_image_directory + image_identifier

            # Storage location for localization maps
            # ..20 classes
            # .. 41x41 grid to interface with original paper
            localization = np.zeros((NUMBER_OF_CLASSES,LOCALIZATION_MAP_SIZE,LOCALIZATION_MAP_SIZE))

            for class_identifier in range(NUMBER_OF_CLASSES):
            # Specity the directory to retrieve the test identifier's annotations
                heatmap = compute_heatmap(gradcam, image_path, target_class=class_identifier)

                # Resize the heatmap to fit the interface
                resized_heatmap = cv2.resize(heatmap,(LOCALIZATION_MAP_SIZE,LOCALIZATION_MAP_SIZE))

                # Activate the heatmap if it is greater than a threshold of 0.20 of maximum intensity
                resized_heatmap = resized_heatmap > 0.20 * np.max(resized_heatmap)

                # Store the localization results for the class
                localization[i,:,:] = resized_heatmap
            
            # Store the new heatmap into the pickle dataset
            data[line] = localization

    # Save the updated pickle file!
    pickle.dump('localization_cues_BY.pickle', data)

    assert(False)
