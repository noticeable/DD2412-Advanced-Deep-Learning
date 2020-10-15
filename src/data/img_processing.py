import cv2
import numpy as np
import os
import torch

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

def load_image(im_path):
    if not os.path.exists(im_path):
        raise FileNotFoundError

    img = cv2.imread(im_path)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    return img

def preprocess_image(img):
    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def prep_im_for_display(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1 + 0.5
    img = np.clip(img, 0, 1)
    img = np.uint8(img * 255)
    return img