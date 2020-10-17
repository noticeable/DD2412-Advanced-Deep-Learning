import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


import torchvision
import torchvision.transforms as transforms
#from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms

from dataset_voc import VOC12
from transform_voc import ToLabel, Relabel


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


datadir = r'C:\Users\reza_062\Documents\Modelon\dataLoader\VOC2012\data'

#loader = DataLoader(VOC12(data_dir))
loader = DataLoader(VOC12(datadir, input_transform, target_transform),
        num_workers= 0, batch_size= 12, shuffle=True)

for epoch in range(1, 4):
    epoch_loss = []

    for step, (images, labels) in enumerate(loader):

        inputs = Variable(images)
        targets = Variable(labels)
        #outputs = model(inputs)