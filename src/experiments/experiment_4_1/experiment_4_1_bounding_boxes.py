import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from time import sleep
from xml.dom import minidom

DEBUG = False
resized_width = 256
resized_height = 256

results = {}

if __name__ == "__main__":

    # Define directories
    VOC2007_annotations_directory = '../../../datasets/VOC/VOC2007/Annotations/'

    # Retrieve the list of files in the directory
    annotated_files =  os.listdir(VOC2007_annotations_directory)
    
    for annotated_file in tqdm(annotated_files):

        # Define file directory
        dom = minidom.parse(VOC2007_annotations_directory + annotated_file)

        # Retrieve the width and height of the original image
        width = int(dom.getElementsByTagName('size')[0].getElementsByTagName('width')[0].firstChild.data)
        height = int(dom.getElementsByTagName('size')[0].getElementsByTagName('height')[0].firstChild.data)

        # Print some useful information!
        print(f'Filename: {annotated_file} \n...width:{width} height:{height}\nBounding Boxes:') if DEBUG else None

        # Only retrieve annotations with one bounding box
        number_of_bounding_boxes = len(dom.getElementsByTagName('object'))

        if number_of_bounding_boxes == 1:
        
            # Extract the bounding boxes provided        
            object_ = dom.getElementsByTagName('object')[0]
            
            # Extract coordinates of each bounding box
            # Extract the class of object being bounded
            classifcation = object_.getElementsByTagName('name')[0].firstChild.data
            
            x_min = int(object_.getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0].firstChild.data)
            y_min = int(object_.getElementsByTagName('bndbox')[0].getElementsByTagName('ymin')[0].firstChild.data)
            x_max = int(object_.getElementsByTagName('bndbox')[0].getElementsByTagName('xmax')[0].firstChild.data)
            y_max = int(object_.getElementsByTagName('bndbox')[0].getElementsByTagName('ymax')[0].firstChild.data)

            print(f'...classification: {classifcation}') if DEBUG else None
            print(f'...original Image size: width: {width}, height: {height}, x_mins: {x_min} x_max:{x_max} y_min:{y_min} y_max:{y_max}') if DEBUG else None

            # resized_x_min = x_min * resized_width   / width
            # resized_x_max = x_max * resized_width   / width
            # resized_y_min = y_min * resized_height  / height
            # resized_y_max = y_max * resized_height  / height
            # print(f'...resized coordinates: width: {resized_width} height: {resized_height} x_min: {resized_x_min} x_max:{resized_x_max} y_min:{resized_x_min} y_max:{resized_y_max}') if DEBUG else None

            # results[annotated_file] = np.array([resized_x_min, resized_y_min, resized_x_max, resized_y_max])
            results[annotated_file] = np.array([x_min, y_min, x_max, y_max, width, height])
            
    with open(r"ground_truth_bounding_boxes_original.pickle", "wb") as output_file:
        pickle.dump(results, output_file)
