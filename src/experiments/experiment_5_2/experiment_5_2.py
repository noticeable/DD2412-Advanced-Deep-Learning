''' Model trust experiment (See section 5.2 from the Grad-CAM paper).
We compare guided back prop and guided grad cam with fine-tuned VGG-16 and AlexNET on Pascal VOC Detetcion Dataset (2012).
We selected 20 images randomly from the test set for this user study.
Consider only images where both models predicted the same as ground truth.
AlexNET is typically the 'worse' model (lower mAP) than VGG-16

Extensions to original experiment:
comparing VGG-16 fine-tuned with pretrained (pretrained performs worse) TODO
'''

from utils.prepare_models import load_fine_trained
from model.gradcam import GradCAM
from model.guided_back_propagation import GuidedBackPropagation

from utils.viz import save_image, get_normalized_image, normalize
import numpy as np
import pandas as pd
import random


def main():
    # Retrieve the model
    print(f'Loading weights for model: {model_name}...')
    model, layer_name = load_fine_trained(model_name, model_weights)

    print(f'Weights loaded!')

    # Instantiate the gradCAM class
    gradcam = GradCAM(model=model, \
                      target_layer=layer_name
                      )

    # load pickle
    data = pd.read_csv(experiment_data_fpath)
    images = data['images'].tolist()
    labels = data['labels'].tolist()
    categories = data['categories'].tolist()


    for i, im_path in enumerate(images):
        label = int(labels[i])
        category = categories[i]

        img_name = f'im_{i}_{category}'
        n = random.randint(0, 1) # if 0, guided back prop is model A and g-grad cam is model B, else guided grad cam is model A and guided back prop is model B

        # get guided grad cam map and guided backprop
        """ Guided Back Propagation """
        # Instantiate guided backpropagation to retrieve the mask
        guided_backpropagation = GuidedBackPropagation(model=model)

        # Obtain the initial image
        input_image = get_normalized_image(im_path)

        # Obtain the image mask by applying gradcam
        guided_backpropagation_mask = guided_backpropagation(input_image)

        # Normalize the guided back propagation mask
        guided_backpropagation_image = normalize(guided_backpropagation_mask)

        # Save the image
        model_for_user = 'model_A' if n == 0 else 'model_B'
        save_image(guided_backpropagation_image, f'../results/experiment_5_2/{model_name}/{img_name}_{model_for_user}_guided_backpropagation.png')

        """ Guided GradCAM"""
        # Reset the guided_backpropagation_mask of the input, so that the input_image tensor can be used again
        input_image.grad.zero_()

        # Obtain the gradcam mask
        mask = gradcam(input_image, target_index=label)

        # Merge the gradcam with the guided back propagation to get guided gradcam results
        guided_gradCAM = guided_backpropagation_mask * mask[..., np.newaxis]

        # Normalize the guided gradCAM results to the [1,255] range
        guided_gradCAM = normalize(guided_gradCAM)

        # save image
        model_for_user = 'model_A' if n == 1 else 'model_B'
        save_image(guided_gradCAM, f'../results/experiment_5_2/{model_name}/{img_name}_{model_for_user}_guided_gradcam.png')



if __name__ == '__main__':

    model_weights = '../weights/vgg16_finetrained.ckpt'
    model_name = 'vgg16'

    # model_weights = '../weights/alexnet_finetrained.ckpt'
    # model_name = 'alexnet'

    experiment_data_fpath = 'experiments/experiment_5_2/experiment_data.csv'

    main()