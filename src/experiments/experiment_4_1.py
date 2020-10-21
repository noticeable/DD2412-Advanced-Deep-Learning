import cv2
import numpy as np

DEBUG = True


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
#   image: cv2.image    : image to draw a bounding box in
# output:
#   bounded_image: cv2.image: image with bounded boxes included
"""
def draw_bounding_box(image):

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
    x, y, w, h = cv2.boundingRect(contour)

    # Print some information if debug is enabled.
    print(f'Number of contours found: {len(contours)}\nCoordinates of bounding box: Upper-left:({x},{y}), Bottom-right:({x+w},{y+h})') if DEBUG else None

    # Draw the bounding box!
    bounded_image = cv2.rectangle(  img=image, \
                                    pt1=(x,y),          # Top left coordinate\
                                    pt2=(x+w,y+h),      # Bottom right coordinate\
                                    color=(75,75,75),   # Color of bounding box\
                                    thickness=3         # thickness of bounding box line\
                                )

    return bounded_image


if __name__ == "__main__":

    # Define path to the image
    image_path = '../../results/test_results/cat_dog_heatmap.png'

    #Read in the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('../../results/experiment_4_1/grayscale.png', image)

    # Binarize the image
    binarized_image = binarize(image, percentage=0.50)

    # Save binarized image
    cv2.imwrite('../../results/experiment_4_1/binarized.png', binarized_image)

    # draw bounding box
    bounded_binarized_image = draw_bounding_box(binarized_image)

    # Save results
    cv2.imwrite('../../results/experiment_4_1/bounded_binarized.png', bounded_binarized_image)