import os
import json
import pandas as pd
from PIL import Image
import argparse
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from data.utils import transform
from data.transform_voc import ToLabel, Relabel

EXTENSIONS = ['.jpg', '.png']

input_transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])

target_transform = transforms.Compose([
    transforms.CenterCrop(256),
    ToLabel(),
    Relabel(255, 21),
])


def load_image(file):
    return Image.open(file, mode='r')

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC2012(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class VOC2007(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    Taken and modified from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
    """

    def __init__(self, data_folder, split, return_bb=False, input_transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param return_bb: if True, returns bounding boxes instead of labels
        """
        self.split = split.upper()
        self.return_bb = return_bb
        self.input_transform = input_transform

        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        self.data_folder = data_folder

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)
        # file with label corresponding to largest object
        self.labels = pd.read_csv(os.path.join(data_folder, self.split + '_largest_item_labels.csv'))['labels'].tolist()

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        if self.return_bb:
            image = Image.open(self.images[i], mode='r')
            image = image.convert('RGB')

            # Read objects in this image (bounding boxes, labels)
            objects = self.objects[i]
            boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
            labels = torch.LongTensor(objects['labels'])  # (n_objects)

            # Apply transformations
            image, boxes, labels = transform(image, boxes, labels, split=self.split)
            return image, boxes
        else:
            image = load_image(self.images[i]).convert('RGB')
            if self.input_transform is not None:
                image = self.input_transform(image)
            label = self.labels[i]
            print('image', self.images[i])
            print('label', label)
            return image, label


    def __len__(self):
        return len(self.images)

    # def collate_fn(self, batch):
    # TODO come back to
    #     """
    #     Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    #
    #     :param batch: an iterable of N sets from __getitem__()
    #     :return: a tensor of images, lists of varying-size tensors of labels.
    #             NOTE: Labels are bounding boxes if self.return_bb is True, otherwise are class labels
    #     """
    #     if self.return_bb:
    #         images = list()
    #         labels = list() # contains Detection labels (class labels) if self.return_bb is false, otherwise returns bounding boxes
    #
    #         for b in batch:
    #             images.append(b[0])
    #             labels.append(b[1])
    #
    #         images = torch.stack(images, dim=0)
    #         return images, labels
    #     else:
    #         return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../datasets/TODO', help='Path to dataset directory. For VOC2012 this is the VOC2012 folder, for VOC2007 this is the VOC2007/loader folder.')
    parser.add_argument('--dataset', type=str, default='../datasets/TODO', help='Dataset to test', choices=['VOC2012', 'VOC2007'])
    parser.add_argument('--batch-size', type=int, default=12)

    args = parser.parse_args()

    datadir = args.data_dir
    dataset = args.dataset
    if dataset == 'VOC2012':
        # VOC2012 (Segmentation) dataloader
        loader = DataLoader(VOC2012(datadir, input_transform, target_transform),
                            num_workers=1, batch_size=args.batch_size, shuffle=True)

        for epoch in range(1, 4):
            epoch_loss = []

            for step, (images, labels) in enumerate(loader):
                inputs = Variable(images)
                targets = Variable(labels)
                # outputs = model(inputs)
    else:
        print('Loading VOC2007 dataset...')
        # VOC2007 (Detection) dataloader
        train_dataset = VOC2007(datadir, split='train', input_transform=input_transform)
        val_dataset = VOC2007(datadir, split='val', input_transform=input_transform)


        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=1,
                                                   pin_memory=True)
        # Batches
        for i, (im, label) in enumerate(train_loader):
            print('Batch', i)
