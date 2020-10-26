import numpy as np
import pickle

DEBUG = False

"""
    Returns True or False if the intersection over union score is greater than 0.50
        :param  bounding_box_1  coodrinates of [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        :param  bounding_box_2  coodrinates of [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        :output boolean         True/False if conditions are met
        :output scalar          Interesction over union score
"""
def calculate_intersection_over_union_score(bounding_box_1, bounding_box_2):
    # Modified from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation/42874377

    """
    NOTE: From PASCAL VOC Challenge
    A predicted bounding box is considered correct if it overlaps more than 50% with a ground-truth
    bounding box, otherwise the bounding box is considered a false positive detection. Multiple 
    detections are penalized. If a system predicts several bounding boxes that overlap with a single 
    ground-truth bounding box, only one prediction is considered correct, the others are considered 
    false positives.
    """

    bounding_box_1_x1, bounding_box_1_y1, bounding_box_1_x2, bounding_box_1_y2 = bounding_box_1
    bounding_box_2_x1, bounding_box_2_y1, bounding_box_2_x2, bounding_box_2_y2 = bounding_box_2

    # Confirm that coordinates are reasonable!
    assert bounding_box_1_x1 < bounding_box_1_x2
    assert bounding_box_1_y1 < bounding_box_1_y2
    assert bounding_box_2_x1 < bounding_box_2_x2
    assert bounding_box_2_y1 < bounding_box_2_y2

    # determine the coordinates of the intersection rectangle
    x_left      = max(bounding_box_1_x1, bounding_box_2_x1)
    y_top       = max(bounding_box_1_y1, bounding_box_2_y1)
    x_right     = min(bounding_box_1_x2, bounding_box_2_x2)
    y_bottom    = min(bounding_box_1_y2, bounding_box_2_y2)

    if x_right < x_left or y_bottom < y_top:
        return False, 0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bounding_box_1_area = (bounding_box_1_x2 - bounding_box_1_x1) * (bounding_box_1_y2 - bounding_box_1_y1)
    bounding_box_2_area = (bounding_box_2_x2 - bounding_box_2_x1) * (bounding_box_2_y2 - bounding_box_2_y1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    intersection_over_union_score = intersection_area / float(bounding_box_1_area + bounding_box_2_area - intersection_area)
    assert intersection_over_union_score >= 0.0
    assert intersection_over_union_score <= 1.0

    if intersection_over_union_score > 0.50:
        return (True, intersection_over_union_score)
    else:
        return (False, intersection_over_union_score)

"""
#   Test case to check IOU is calculating correctly
#   :param  None
"""
def test_case():
    # Bounding box : [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    bounding_box_1 = np.array([0,0,2,2])
    bounding_box_2 = np.array([1,0,3,2])

    results, intersection_over_union_score = calculate_intersection_over_union_score(bounding_box_1, bounding_box_2)

    print(f'>0.50? {results}.  intersection_over_union_score : {intersection_over_union_score}')


"""
#   Computes the accuracy of the gradCAM bounding boxes with respect to VOC2007.
"""
def score_gradCAM():
    # Define paths to files
    ground_truth_resized_bounding_boxes_path = './ground_truth_bounding_boxes.pickle'
    gradCAM_resized_bounding_boxes_path = './gradCAM_bounding_boxes.pickle'
    VOC2007_images_directory = '../../../datasets/VOC/VOC2007/JPEGImages/'

    # Read in the ground truth bounding boxes
    with open(ground_truth_resized_bounding_boxes_path,'rb') as opened_file:
        ground_truth_bounding_boxes = pickle.load(opened_file)

    # Read in the gradCAM bounding boxes
    with open(gradCAM_resized_bounding_boxes_path,'rb') as opened_file:
        gradCAM_bounding_boxes = pickle.load(opened_file)


    # Calculate the number of images
    number_of_images = len(ground_truth_bounding_boxes.keys())
    print(f'Number of Images: {number_of_images}') if DEBUG else None# 3770

    # Get a list of the identifiers in the dictionaries
    image_identifiers = list(ground_truth_bounding_boxes.keys())

    # Accumulate amount of correctly bounded images
    accepted_bounding_boxes = 0

    # Iterate through every image
    for image_identifier in image_identifiers:
        # Retrieve the gradCAM and ground truth bounding boxes for the image
        gradCAM_bounding_box = gradCAM_bounding_boxes[image_identifier]
        ground_truth_bounding_box = ground_truth_bounding_boxes[image_identifier]

        # Compute the IOU score and 
        results, intersection_over_union_score= calculate_intersection_over_union_score(gradCAM_bounding_box, ground_truth_bounding_box)

        # rename the variable for readability
        intersection_over_union_is_greater_than_fifty_percentage = results

        # Increment the score if correctly bounded
        if intersection_over_union_is_greater_than_fifty_percentage:
            accepted_bounding_boxes += 1


    print(f'Out of {number_of_images} images, {accepted_bounding_boxes} were correctly bounded.  This results in {accepted_bounding_boxes/number_of_images} images correctly bounded by gradCAM.')



if __name__ == "__main__":
    # test_case()

    score_gradCAM()