import torch
import cv2
import numpy as np
from torchvision import models

from utils.viz import save_image, get_image, get_normalized_image, normalize, get_heatmap

class GradCAM:

    def __init__(self, model, target_layer=None):
        # Model being investigated
        self.model = model

        # Dictionary to store gradient and activation values
        self.gradients = {}
        self.activations = {}

        # Defining the callable function that will be used to store the gradients
        # as computed in the backward pass and stores it in the gradient dictionary
        def fn_backwards(module, gradient_input, gradient_output):
            self.gradients['value'] = gradient_output[0]

        # Defining the callable function that will be used to store the activations
        # as computed in the forward pass in the activation dictionary
        def fn_forward(module, input, output):
            self.activations['value'] = output

        # Map the forward and backward callable function to 
        # the callback hooks at the deisred layer in order to retrieve
        # the gradients and activations
        target_layer.register_forward_hook(fn_forward)
        target_layer.register_backward_hook(fn_backwards)

        # Set the model into evaluation mode
        self.model.eval()


    """ ReLU
    # inputs:
    #   input:  Tensor: tensor to apply ReLU activations to
    #  outputs:
    #   output: Tensor: tensor post ReLU activation
    """
    def ReLU(self, input):
        output = np.maximum(np.sum(input,axis=0), 0)
        return output

    """
    # Retrieves the heatmap mask as per the gradCAM paper
    # inputs:
    #   gradients:      Tensor: gradients flowing backwards
    #   activations:    Tensor: outputs of the forward pass
    #   image_height:   scalar: number of pixels in the rows
    #   image_width:    scalar: number of pixels in the columns
    # outputs:
    #   gcam_map:       np.matrix: gradCAM mask for the heatmap
    """
    def retrieve_heatmap(   self,   \
                            gradients, \
                            activations, \
                            image_height, \
                            image_width, \
                            ):
        # Apply global average pooling by applying two views 1st. averaging 2nd.pooling
        # NOTE: After squeezing and taking the mean (i.e. global average pooling), alpha is now a one dimensional vector
        alpha = np.mean(gradients.squeeze().data.numpy(), axis=(1,2))

        # Activations is a three dimensional vector 
        activations = activations.squeeze().data.numpy()

        """
        # Paper page 5: Perform a weighted combination of forward activation maps, followed by a Relu to obtain
        # L_{Grad-CAM}^{c} = ReLU( \sum_{k} a_{k}^{c} A^{k} )
        #
        # ReLu (default activation) is applied to the linear combination of maps because we are only
        # interested in the features that have a positive influence on the class of interest
        # i.e. pixel whose intensity should be increased in order to increase y^{c}.
        """
        gcam_map = self.ReLU(activations * alpha[:, np.newaxis, np.newaxis])

        """
        # Paper page 5: Notice that this results in a coarse heatmap of the same size as the convolutional
        # feature maps.
        """
        gcam_map -= np.min(gcam_map)
        gcam_map /= np.max(gcam_map)
        gcam_map = cv2.resize(gcam_map, (image_height, image_width))


        return gcam_map


    """
    # Creates one hot encoding for the target class
    # inputs:
    #   predictions: tensor: output of the model's forward pass
    #   target_index: scalar: desired class for investigation
    # outputs:
    #   one_hot_encoding: tensor: one hot encoding of target class
    """
    def get_target(self, predictions, target_index=None):
        if target_index == None:
            target_index = np.argmax(predictions.data.numpy())
        
        target = predictions[0][target_index]

        return target


    """
    Defines the forward method for the model which retrieves the 
        heat map for gradcam

        inputs:
            input: matrix: input image of the matrix
            target_index: scalar: the index of the class that should be investigated
        output:
            gcam_map: matrix: gradcam heat map for visualization
    """
    def forward(self, input, target_index = None):
        # Retrieve the dimensions of the input
        _, _, image_height, image_width = input.size() 

        # Reset the gradients in the model to 0 prior to conducting backward pass
        self.model.zero_grad()

        # Apply a forward pass to the model
        predictions = self.model(input)

        # Obtain the target class for back-propagation
        target = self.get_target(predictions, target_index = target_index)

        # Conduct the backward pass
        target.backward()
        
        # Make activations and alpha the same dimensions by adding new axis to the one dimensional alpha
        gcam_map = self.retrieve_heatmap(   self.gradients['value'], \
                                            self.activations['value'], \
                                            image_height, \
                                            image_width)

        return gcam_map


    def __call__(self, input, target_index=None):
        return self.forward(input, target_index=target_index)


if __name__ == '__main__':
    # Test on Resnet50
    model = models.resnet50(pretrained=True)

    # Instantiate the gradCAM class
    gradcam = GradCAM(  model=model, \
                        target_layer = model.layer4
                        )

    # Obtain pre-normalized image
    image = get_image('../../datasets/test_images/cat_dog.png')

    # Obtain normalized image
    normalized_image = get_normalized_image('../../datasets/test_images/cat_dog.png')

    # Obtain the image mask by applying gradcam
    mask = gradcam(normalized_image)

    # Obtain the result of merging the mask with the torch image
    heatmap = get_heatmap(mask)

    # Merge the heatmap with 
    cam = heatmap + np.float32(image)

    # Normalize the merged results
    cam = normalize(cam)

    # Save the results!
    save_image(cam, '../../results/test_results/cat_dog_gradcam.png')
