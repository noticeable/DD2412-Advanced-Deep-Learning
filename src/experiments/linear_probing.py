"""
# Linear Probing Experiment
# Goal: 
#   Investigate the heatmap generated from each layer of the network
#   and investigate if the generally accepted wisdom that -
#   the deepest layer captures the most complexity and provides 
#   the most explanatory value - is reasonable.
"""

import numpy as np

from model.gradcam import GradCAM
from model.guided_back_propagation import GuidedBackPropagation
from utils.viz import save_image, get_image, get_normalized_image, normalize, get_heatmap
from utils.prepare_models import get_model


"""
# Disable warnings of low quality image
"""

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

"""
# Captures and handles processing of unusual layers
# input:
#   module_name:    string: name of the network's module under investigation
# output:
#   _:              boolean: true/false reflecting unusual or not
"""
def unusual_case_identified(module_name, model_name="resnet50"):

    # Define unusual layer conditions in pytorch resnet architecture
    if model_name == "resnet50":
        
        # Condition 1: Module is unnamed
        if module_name.strip() == "":
            return True

        # Condition #2: Relu layer - distorts image shape and requires reshaping to meet gradcam dimensions
        elif "relu" in module_name:
            return True
        
        # Condition #3: Avg pool layer - distorts image shape and requires reshaping to meet gradcam dimensions
        elif "avgpool" in module_name:
            return True

        elif "fc" == module_name:
            return True

    elif model_name == "vgg16":
        pass
    elif model_name == "inception_V3" or model_name == "googlenet":
        pass
    elif model_name == "alexnet":
        pass

    return False



if __name__ == '__main__':
    # Define requested model; please see prepare_models.py for a list of valid models
    model_name = 'resnet50'
    
    # Retrieve the model
    model, _ = get_model(model_name)

    # Iterate through all the layers
    for module_name, module in model.named_modules():

        # NOTE: model.named_modules() can return an empty module name...
        if unusual_case_identified(module_name):
            print(f"....Skipping unusual layer...{module_name}...")
            continue

        # Print information!
        print(f"Investigating {model_name}'s {module_name} module")

        # Instantiate the gradCAM class
        gradcam = GradCAM(  model = model, \
                            target_layer = module
                            )

                                # Obtain pre-normalized image
        image = get_image('../../datasets/test_images/cat_dog.png')

        # Obtain normalized image
        normalized_image = get_normalized_image('../../datasets/test_images/cat_dog.png')

        # Obtain the image mask by applying gradcam
        mask = gradcam(normalized_image)

        # Obtain the result of merging the mask with the torch image
        heatmap = get_heatmap(mask)

        # Save the heatmap!
        save_image(heatmap, f'../../results/linear_probing/heatmap/cat_dog_heatmap_{model_name}_{module_name}.png')

        # Merge the heatmap with 
        cam = heatmap + np.float32(image)

        # Normalize the merged results
        cam = normalize(cam)

        # Save the results!
        save_image(cam, f'../../results/linear_probing/gradcam/cat_dog_gradcam_{model_name}_{module_name}.png')


