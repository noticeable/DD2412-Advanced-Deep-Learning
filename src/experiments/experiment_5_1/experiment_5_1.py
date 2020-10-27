''' Class Discrimination Experiment
We compare guided backprop and guided grad cam with fine-tuned VGG-16 on Pascal VOC Detetcion Dataset (2012)
to see if grad cam improves ability to discriminate between classes.
We selected 20 images randomly from the test set for this user study and ask the user which shows a better visualisation
out of guided grad cam and guided back prop.
These images contain 2 categories. We randomly choose one of the categories to visualise map for.
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
    labels = data['labels'].apply(lambda x: x[1:-1].split(',')).tolist()
    categories = data['categories'].apply(lambda x: x[1:-1].split(',')).tolist()
    main_category_index = data['main_category_index'].tolist()

    for i, im_path in enumerate(images):
        label_set = labels[i]
        category_set = categories[i]
        # n = random.randint(0,1)
        n = int(main_category_index[i])
        label = int(label_set[n])

        category = category_set[n]
        second_category = category_set[1-n]
        img_name = f'im_{i}_{category}_other_category_{second_category}'

        # get guided grad cam map and guided backprop

        """ Guided Back Propagation """
        # Instantiate guided backpropagation to retrieve the mask
        guided_backpropagation = GuidedBackPropagation(model=model)

        # Obtain the initial image
        input_image = get_normalized_image(im_path)

        # Obtain the image mask by applying gradcam
        guided_backpropagation_mask = guided_backpropagation(input_image, label)

        # Normalize the guided back propagation mask
        guided_backpropagation_image = normalize(guided_backpropagation_mask)

        # Save the image
        save_image(guided_backpropagation_image, f'../results/experiment_5_1/{model_name}/{img_name}_guided_backpropagation.png')

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
        save_image(guided_gradCAM, f'../results/experiment_5_1/{model_name}/{img_name}_guided_gradcam.png')



if __name__ == '__main__':

    # model_weights = '../weights/vgg16_finetrained.ckpt'
    # model_name = 'vgg16'

    model_weights = '../weights/alexnet_finetrained.ckpt'
    model_name = 'alexnet'

    experiment_data_fpath = 'experiments/experiment_5_1/experiment_data.csv'

    main()
