import cv2
import numpy as np
import pickle
from tqdm import tqdm

DEBUG = False
image_size = 256

from utils.viz import save_image, get_image, get_normalized_image, normalize, get_heatmap


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


""" Binarizes an image such that pixel greater than 15% of the image's maximum intensity are set to 1; 0 otherwise.
# input: 
#   image:          cv2.image:  cv2 image matrix to be worked on
#   percentage:     scalar:     threshold percentage to determine value
# output:
#   binarized_image:cv2_image:  binarized cv2 image matrix
"""
def binarize(image, percentage=0.15):

    # Retrieve the maximum intensity
    maximum_intensity = image.max()

    # Define the threshold as a percentage of the maximum intensity
    threshold_value = percentage * maximum_intensity

    # Display some information if DEBUG is enabled.
    print(f'Maximum intensity: {maximum_intensity} \nThreshold value: {threshold_value}') if DEBUG else None

    # Binarize the image at the threshold value
    _, binarized_image = cv2.threshold( src=image, \
                                        thresh=threshold_value,\
                                        maxval=255, \
                                        type=cv2.THRESH_BINARY\
                                        )
    return binarized_image


"""
# Draws a bounding box around the largest contiguous cluster of binarized pixels
# inputs: 
#   :param  scalar  x   top left x position
#   :param  scalar  y   top left y position
#   :param  scalar  w   width
#   :param  scalar  h   height
# output:
#   bounded_image: cv2.image: image with bounded boxes included
"""
def draw_bounding_box(x,y,width,height, image, color=(255,0,0)):

    # Draw the bounding box!
    bounded_image = cv2.rectangle(  img=image, \
                                    pt1=(x,y),          # Top left coordinate\
                                    pt2=(x+width,y+height),      # Bottom right coordinate\
                                    color=color,   # Color of bounding box\
                                    thickness=3         # thickness of bounding box line\
                                )

    return bounded_image


"""
# Computes the coordinates of the bounding box
#   :param  matrix  image   image to investigate   
#   :output scalar  x       top left x position
#   :output scalar  y       top left y position
#   :output scalar  w       width
#   :output scalar  h       height
"""
def compute_bounding_box(image):
    # Finds the largest contiguous block in the image
    contours, hierarchy = cv2.findContours( image=image, \
                                            mode=cv2.RETR_TREE,\
                                            method=cv2.CHAIN_APPROX_SIMPLE \
                                            )

    # If there are multiple contiguous blocks, select the largest
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
    else:
        contour = contours[0]

    # Determine the coordinates of the minimum bounding box around the contour
    x, y, width, height = cv2.boundingRect(contour)

    # Print some information if debug is enabled.
    print(f'Number of contours found: {len(contours)}\nCoordinates of bounding box: Upper-left:({x},{y}), Bottom-right:({x+width},{y+height})') if DEBUG else None

    return x, y, width, height


def test_case():

    # Define path to the image
    image_path = '../../../results/test_results/cat_dog_heatmap.png'

    #Read in the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('../../../results/experiment_4_1/grayscale.png', image)

    # Binarize the image
    binarized_image = binarize(image, percentage=0.50)

    # Save binarized image
    cv2.imwrite('../../../results/experiment_4_1/binarized.png', binarized_image)

    # Get the positions of the bounding box
    x, y, width, height = compute_bounding_box(binarized_image)

    # draw bounding box
    bounded_binarized_image = draw_bounding_box(x=x, \
                                                y=y, \
                                                width=width, \
                                                height=height, \
                                                image=image
                                                )

    # Save results
    cv2.imwrite('../../../results/experiment_4_1/bounded_binarized.png', bounded_binarized_image)

"""
# Computes and stores the bounding box coordinates computed using GradCAM
# :param    
"""
def process_VOC2007(generate_bounding_box_images=False):
    # Dictionary to store the results
    gradCAM_bounding_boxes = {}

    # Define directories
    ground_truth_resized_bounding_boxes_path = './ground_truth_bounding_boxes_original.pickle'
    VOC2007_images_directory = '../../../datasets/VOC/VOC2007/JPEGImages/'
    gradCAM_heatmaps_directory = '../../../results/experiment_4_1/heatmaps/'

    # Read the ground truth file to get list of images to process
    with open(ground_truth_resized_bounding_boxes_path,'rb') as opened_file:
        ground_truth_bounding_boxes = pickle.load(opened_file)

    print(f'Number of Images: {len(ground_truth_bounding_boxes.keys())}') if DEBUG else None # 3770

    # Iterate through every image and compute bounding boxes
    for image_identifier in tqdm(list(ground_truth_bounding_boxes.keys())):

        # Strip the .xml from the identifier
        image_identifier = image_identifier.strip('.xml')

        # Define the image path
        image_path = VOC2007_images_directory + image_identifier + '.jpg'

        # Define the heatmap path
        heatmap_path = gradCAM_heatmaps_directory + image_identifier + '.png'

        # Read in the image
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Read in the heatmap
        original_heatmap = cv2.imread(heatmap_path, cv2.IMREAD_COLOR)

        # Convert to grayscale
        heatmap = cv2.cvtColor(original_heatmap, cv2.COLOR_BGR2GRAY)

        # Binarize the image
        binarized_heatmap = binarize(heatmap, percentage=0.50)

        # Get the positions of the bounding box
        x, y, width, height = compute_bounding_box(binarized_heatmap)

        # Store the results in the dictioary
        gradCAM_bounding_boxes[image_identifier] = np.array([x, y, x + width, y + height])

        # Draw bounding box images
        if generate_bounding_box_images:

            # Extract information from dictionary
            ground_truth_x1, \
            ground_truth_y1, \
            ground_truth_x2, \
            ground_truth_y2, \
            original_width, \
            original_height = ground_truth_bounding_boxes[image_identifier + '.xml']

            # Cast to int
            ground_truth_x1 = int(ground_truth_x1)
            ground_truth_x2 = int(ground_truth_x2)
            ground_truth_y1 = int(ground_truth_y1)
            ground_truth_y2 = int(ground_truth_y2)
            original_width = int(original_width)
            original_height = int(original_height)
            ground_truth_width = int(ground_truth_x2 - ground_truth_x1)
            ground_truth_height = int(ground_truth_y2 - ground_truth_y1)

            cv2.imwrite(f'../../../results/experiment_4_1/binarized/{image_identifier}.png', binarized_heatmap)


            # draw bounding box for gradcAM coordinates
            bounded_binarized_image = draw_bounding_box(x       = x, \
                                                        y       = y, \
                                                        width   = width, \
                                                        height  = height, \
                                                        image   = original_image, \
                                                        color   = (255,0,0)
                                                        )

            # draw bounding box for ground truth
            bounded_binarized_image = draw_bounding_box(x       = ground_truth_x1, \
                                                        y       = ground_truth_y1, \
                                                        width   = ground_truth_width, \
                                                        height  = ground_truth_height, \
                                                        image   = bounded_binarized_image, \
                                                        color   = (255,255,255)
                                                        )

            # Save results
            cv2.imwrite(f"../../../results/experiment_4_1/bounding_box/{image_identifier}.png", bounded_binarized_image)

    with open(r"gradCAM_bounding_boxes.pickle", "wb") as output_file:
        pickle.dump(gradCAM_bounding_boxes, output_file)

if __name__ == "__main__":
    # test_case()
    process_VOC2007(generate_bounding_box_images=True)