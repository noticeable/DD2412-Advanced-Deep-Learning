import cv2
import torch
import numpy as np
import PIL

from torchvision import models, transforms
from torch.nn import functional as F
from torch.nn import Module

from utils.viz import viz_gradcam

class GradCAM:

    def __init__(self, model: Module, target_layer: Module):
        # Model being investigated
        self.model = model

        # Dictionary to store gradient and activation values
        self.gradients = {}
        self.activations = {}

        # Activation function to use in gradcam
        self.cam_activation = F.relu

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


    def retrieve_heatmap(   self,   \
                            gradients, \
                            activations, \
                            image_height, \
                            image_width, \
                            ):

        # Retrieve the dimensions of the gradients
        batch_size, number_of_channels, height, width = gradients.size()

        # Apply global average pooling by applying two views 1st. averaging 2nd.pooling
        alpha = gradients.view(batch_size, number_of_channels, -1).mean(2).view(batch_size,number_of_channels, 1, 1) 
        """
        # Paper page 5: Perform a weighted combination of forward activation maps, followed by a Relu to obtain
        # L_{Grad-CAM}^{c} = ReLU( \sum_{k} a_{k}^{c} A^{k} )
        #
        # ReLu (default activation) is applied to the linear combination of maps because we are only
        # interested in the features that have a positive influence on the class of interest
        # i.e. pixel whose intensity should be increased in order to increase y^{c}.
        """
        gcam_map = (alpha * activations).sum(1, keepdim=True)
        gcam_map = self.cam_activation(gcam_map)

        """
        # Paper page 5: Notice that this results in a coarse heatmap of the same size as the convolutional
        # feature maps.
        """
        gcam_map = F.interpolate(gcam_map, size=(image_height, image_width), mode='bilinear', align_corners=False)
        gcam_map = (gcam_map - gcam_map.min()).div(gcam_map.max()-gcam_map.min()).data

        return gcam_map


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

        # Apply a forward pass to the model
        predictions = self.model(input)

        if target_index is None:
            # If we are investigating the most likely output
            # retrieve the output associated with that class
            score = predictions[:, predictions.max(1)[-1]].squeeze()
        else:
            # If we are investigating a specific class, retrieve the output
            # associated with that class
            score = predictions[:, target_index].squeeze()

        # Reset the gradients in the model to 0 prior to training
        self.model.zero_grad()

        # Conduct the backward pass
        score.backward()

        # Retireve the heatmap
        gcam_map = self.retrieve_heatmap( self.gradients['value'], self.activations['value'], image_height, image_width)

        return gcam_map


    def __call__(self, input, target_index=None):
        return self.forward(input, target_index=target_index)


if __name__ == '__main__':
    # Test on Resnet50
    model = models.resnet50(pretrained=True)

    # Model.to('cpu').eval() extracts the configuration from the model
    gradcam = GradCAM(  model=model.to('cpu').eval(), \
                        target_layer = model.layer4
                        )

    # Obtain the initial image
    image = PIL.Image.open('../../datasets/test_images/cat_dog.png')

    # Convert the image into torch format
    torch_image = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                                ])(image).to('cpu')

    # Normalize the image
    normalized_image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_image)[None]

    # Obtain the image mask by applying gradcam
    mask = gradcam(normalized_image)

    # Obtain the result of merging the mask with the torch image
    viz_gradcam(torch_image, mask, output_file="../../results/test_results/cat_dog_gradcam.jpeg")