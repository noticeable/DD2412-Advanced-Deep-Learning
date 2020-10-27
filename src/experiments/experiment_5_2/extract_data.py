# utils for experiments
from data.datasets import VOC2007
from utils.prepare_models import load_fine_trained
from model.lightning_model import input_transform
import torch
from torch.utils.data import Dataset
import json
import pandas as pd

def _get_pred(model, im):
    pred = model(im)
    pred = model.softmax(pred)
    pred = torch.argmax(pred, axis=1)
    return pred

def get_images_from_lightning(model1, model2, csv_fpath='experiments/experiment_5_2/experiment_data.csv', n_images=20, check_pred=True):
    # randomly select n_images
    datadir = '../datasets/VOC/VOC2007/loader/'
    test_dataset = VOC2007(datadir, split='train', input_transform=input_transform, return_im_path=True)

    dataloader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=1)
    images = []
    labels = []
    categories = []
    n = 0

    # get label2category map to save category with image
    with open('../weights/label_map.json', 'r') as f:
        category2labelmap = json.load(f)
        label2categorymap = {category2labelmap[category]: category for category in category2labelmap}

    # loop through shuffled data (pick image at random)
    for i, (im_path, im, label) in enumerate(dataloader):
        if check_pred:
            pred1 = _get_pred(model1, im)
            use_image = (pred1 == label).sum().item()
            if use_image: # check second model is also correct
                pred2 = _get_pred(model2, im)
                use_image = (pred2 == label).sum().item() # already know pred1 is correct
        else:
            use_image = True

        if use_image:
            images.append(im_path[0]) # append original image as this is what we want to save
            label = label.item()
            category = label2categorymap[label]
            categories.append(category)
            labels.append(label)
            n += 1
        print('Using image', i, ':', use_image==1)
        if n == n_images:
            break

    # save to pickle file
    data = {'categories': categories, 'labels': labels, 'images': images}
    df = pd.DataFrame(data)
    df.to_csv(csv_fpath)



if __name__ == '__main__':
    model_weights = '../weights/vgg16_finetrained.ckpt'
    model_name = 'vgg16'
    vgg_model, _ = load_fine_trained(model_name, model_weights)

    model_weights = '../weights/alexnet_finetrained.ckpt'
    model_name = 'alexnet'
    alexnet_model, _ = load_fine_trained(model_name, model_weights)

    experiment_data_fpath = 'experiments/experiment_5_2/experiment_data.csv'

    get_images_from_lightning(vgg_model, alexnet_model, experiment_data_fpath)
