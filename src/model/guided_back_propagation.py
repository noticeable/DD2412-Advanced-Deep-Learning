import torch
import numpy as np
from torchvision import models

from model.gradcam import GradCAM
from utils.viz import save_image, get_normalized_image, normalize
from utils.prepare_models import get_model

class GuidedBackPropagation:

    def __init__(self, model):
        # Store the model in the class
        self.model = model

        # Apply the customized ReLU for backward pass
        # No changes are needed for the forward pass, as it behaves the same
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(self.fn_backward)
        
        # Put the model in evaluation mode
        self.model.eval()


    """
    # Guided Backprogation ReLU
    # The standard ReLU is adjusted allow positive guided_backpropagation_mask to pass.  Therefore in the backpass, only the following 
    # guided_backpropagation_mask can flow through to the next layer
    #   1. (standard)   guided_backpropagation_mask of nodes that were positive in the forward pass
    #   2. (new)        guided_backpropagation_mask of nodes that are positive in the backward pass
    # input:
    #   module: torch.module: the module in the model under investigation
    #   gradient_input:     tensor: tensor of guided_backpropagation_mask coming from the forward pass
    #   gradient_output:    tensor: tensor of guided_backpropagation_mask coming from the backward pass 
    # output:
    #   _                   tensor: tensor of guided_backpropagation_mask that satisfy condition 1 & 2 above
    """
    def fn_backward(self, module, gradient_input, gradient_output):
        return torch.clamp(gradient_input[0], min = 0.0),


    """
    # Conducts guided back propagation and extracts the gradients
    # inputs:
    #   inputs:         tensor: image under investigation
    #   target_index:   scalar: class to investigate
    # outputs:
    #   gradients:      tensor: gradients to propagate backwards
    """
    def backward(self, inputs, target_index=None):
        # Set guided_backpropagation_mask to zero in the model
        self.model.zero_grad()

        # Conduct a forward pass of the model
        predictions = self.model(inputs)

        # If no target class is defined, investigate the model's most probable class
        if target_index == None:
            target_index = np.argmax(predictions.data.numpy())
        
        # Define the target class to investigate in the back propagation
        target_class = predictions[0][target_index]

        # Conduct the backward pass
        target_class.backward()

        # obtain the gradients
        gradients = inputs.grad[0].data.numpy()

        # Reorganize the gradients to [224,224,3]
        gradients = np.transpose(gradients,(1,2,0))
        
        return gradients

    def __call__(self, inputs, target_index=None):
        return self.backward(inputs, target_index = target_index)


if __name__ == '__main__':
    # Retrieve the model
    # please see prepare_models.py for a list of valid models
    # Dated: 21.10.2020; valid_models = ['resnet50','vgg16','googlenet','inception_v3', 'alexnet']
    model, layer_name = get_model('resnet50')

    # Instantiate gradcam
    gradcam = GradCAM(  model=model, \
                        target_layer = layer_name
                        )

    """ Guided Back Propagation """
    # Instantiate guided backpropagation to retrieve the mask
    guided_backpropagation = GuidedBackPropagation(model=model)

    # Obtain the initial image
    input_image = get_normalized_image('../../datasets/test_images/cat_dog.png')

    # Obtain the image mask by applying gradcam
    guided_backpropagation_mask = guided_backpropagation(input_image)

    # Normalize the guided back propagation mask
    guided_backpropagation_image = normalize(guided_backpropagation_mask)

    # Save the image!
    save_image(guided_backpropagation_image, '../../results/test_results/cat_dog_guided_backpropagation.png')

    """ Guided GradCAM"""

    # Reset the guided_backpropagation_mask of the input, so that the input_image tensor can be used again
    input_image.grad.zero_()

    # Obtain the gradcam mask
    mask = gradcam(input_image)

    # Merge the gradcam with the guided back propagation to get guided gradcam results
    guided_gradCAM = guided_backpropagation_mask * mask[..., np.newaxis]

    # Normalize the guided gradCAM results to the [1,255] range
    guided_gradCAM = normalize(guided_gradCAM)

    # save image!
    save_image(guided_gradCAM, '../../results/test_results/cat_dog_guided_gradcam.png')
