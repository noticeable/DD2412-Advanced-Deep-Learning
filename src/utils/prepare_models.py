import torch
from torchvision import models

# List of supported models
# Dated: 21.10.2020
valid_models = ['resnet50','vgg16','googlenet','inception_v3', 'alexnet']

"""
# Retrieves the requested model and loads the weights if a path to the weights is provided
# If no weights are provided, the pre-trained weights will be loaded.
# If no layer name is provided, the function will look for the last convolutional 2d layer
# inputs:
#   model_name:     string:         name of the model to be requested
#   weight_path:    string:         path to the fine-tuned/customized weights
#   layer_name:     string:         name of the layer where the hooks for gradcam/guided-gradcam should be loaded
# outputs:
#   model:          torch.model:    requested model with weights loaded
#   layer_name:     string:         name of the layer which gradcam/guided-gradcam should be applied
"""
def get_model(  model_name = "resnet50", \
                weight_path = None, \
                layer_name = None \
                ):

    # Raise an error if an unsupported model is requested
    if model_name not in valid_models:
        raise ValueError(f'\n\nERROR: Invalid model requested.  Please select from the supported models in [{", ".join(valid_models)}].')

    # If weights are not provided, use the pretrained weights
    pretrained = weight_path is None

    # Retrieve the models
    if model_name == 'resnet50':
        # model = models.resnet50(pretrained=pretrained)
        model = models.segmentation.fcn_resnet50(pretrained=pretrained)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
    elif model_name in ['googlenet', 'inception_v3']:
        # NOTE: googLeNet was known as inception_v1.  Thus, we extend the usage to the contemporary inception_v3
        model = models.inception_v3(pretrained=pretrained)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=pretrained)

    # Load weights into the model
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))

    """ Retreiving Layer for investigation """
    layer_name = get_layer_name(model_name, model)

    return model, layer_name

def get_layer_name(model_name, model):

    default =   {   'resnet50'      : 'layer4',\
                    'vgg16'         : 'features_29',\
                    'alexnet'       : 'features_11',\
                    'inception_v3'  : 'Mixed_7c'    # AKA GoogLeNet
                }

    # Raise an error if an unsupported model is requested
    if model_name not in valid_models:
        raise ValueError(f'\n\nERROR: Invalid model requested.  Please select from the supported models in [{", ".join(valid_models)}].')

    if model_name == 'resnet50':
        layer_name = model.layer4
    elif model_name == 'vgg16':
        layer_name = model.features
    elif model_name in ['googlenet', 'inception_v3']:
        layer_name = model.Mixed_7c
    elif model_name == 'alexnet':
        layer_name = model.features

    

    return layer_name