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
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
        guided_backpropagation_mask = guided_backpropagation(input_image, label)

        # Normalize the guided back propagation mask
        guided_backpropagation_image = normalize(guided_backpropagation_mask)

        # Save the image
        img_dir = f'../results/experiment_5_2/{img_name}/guided_back_prop'
        Path(img_dir).mkdir(parents=True, exist_ok=True)

        save_image(guided_backpropagation_image, f'{img_dir}/{model_name}.png')

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
        img_dir = f'../results/experiment_5_2/{img_name}/guided_gradcam'
        Path(img_dir).mkdir(parents=True, exist_ok=True)

        save_image(guided_gradCAM, f'{img_dir}/{model_name}.png')


def select_random_model(dir):
    img_dirs = os.listdir(dir)
    img_dirs = [dir + '/' + img_dir for img_dir in img_dirs]
    for img_dir in img_dirs:
        n = random.randint(0,1)
        if n == 0:
            vgg16 = 'modelA'
            alexnet = 'modelB'
        else:
            vgg16 = 'modelB'
            alexnet = 'modelA'

        guided_backprop_dir = f'{img_dir}/guided_back_prop'
        os.rename(f'{guided_backprop_dir}/vgg16.png', f'{guided_backprop_dir}/vgg_16_{vgg16}.png')
        os.rename(f'{guided_backprop_dir}/alexnet_old.png', f'{guided_backprop_dir}/alexnet_{alexnet}.png')

        guided_gradcam_dir = f'{img_dir}/guided_gradcam'
        os.rename(f'{guided_gradcam_dir}/vgg16.png', f'{guided_gradcam_dir}/vgg16_{vgg16}.png')
        os.rename(f'{guided_gradcam_dir}/alexnet_old.png', f'{guided_gradcam_dir}/alexnet_{alexnet}.png')





def plot_on_same_grid(dir):
    img_dirs = os.listdir(dir)
    img_dirs = [dir + '/' + img_dir for img_dir in img_dirs]

    def _save_img(directory):
        files = os.listdir(directory)
        files = [f for f in files if f.endswith('.png')]

        modelA_index = 0 if 'modelA' in files[0] else 1
        modelB_index = 1 - modelA_index

        modelA_path = f'{directory}/{files[modelA_index]}'
        modelB_path = f'{directory}/{files[modelB_index]}'

        modelA_im = mpimg.imread(modelA_path)
        modelB_im = mpimg.imread(modelB_path)

        rows = 1
        cols = 2
        axes = []
        fig = plt.figure()
        images = [modelA_im, modelB_im]
        titles = ['Robot A', 'Robot B']

        for a in range(rows * cols):
            axes.append(fig.add_subplot(rows, cols, a + 1))
            subplot_title = titles[a]
            axes[-1].set_title(subplot_title)
            plt.imshow(images[a])


        fig.tight_layout()

        axes[0].axis('off')
        axes[1].axis('off')

        # deal with margins
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=1, wspace=0.1)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f'{directory}/models.png', bbox_inches = 'tight',
    pad_inches = 0)


    for img_dir in img_dirs:
        if 'im_12' in img_dir:
            guided_backprop_dir = f'{img_dir}/guided_back_prop'
            _save_img(guided_backprop_dir)

            guided_gradcam_dir = f'{img_dir}/guided_gradcam'
            _save_img(guided_gradcam_dir)

if __name__ == '__main__':

    # model_weights = '../weights/vgg16_finetrained.ckpt'
    # model_name = 'vgg16_old'

    model_weights = '../weights/alexnet_finetrained.ckpt'
    model_name = 'alexnet_old'
    #
    experiment_data_fpath = 'experiments/experiment_5_2/experiment_data.csv'

    # main()
    # select_random_model('../results/experiment_5_2')
    plot_on_same_grid('../results/experiment_5_2')