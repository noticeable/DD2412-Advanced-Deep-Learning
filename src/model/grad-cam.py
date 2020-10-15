import cv2
import numpy as np
import torch
from torchvision import models
from torch.nn import functional as F

class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs.append(x)
        return outputs, x


class _BaseCAM:
    def __init__(self, model, feature_module, target_layer_names):
        self.model = model
        self.feature_module = feature_module
        self.target_layer_names = target_layer_names
        self.gradients = []
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layer_names)

    def extractor(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_activations_output(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layer_names:
                x.register_hook(self.save_gradient) # save gradient every time gradient is calculated
                outputs += [x]

        return outputs, x


class GradCAM(_BaseCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        super(GradCAM, self).__init__(model, feature_module, target_layer_names)
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.cam_activation = F.relu



    def forward(self, input):
        return self.model(input)

    def one_hot_encode(self, output, target_index):
        one_hot = torch.zeros_like(output)
        # one_hot = torch.zeros_like(output, requires_grad=True)
        one_hot[0, target_index] = 1

        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        return one_hot


    def __call__(self, input, target_index=None):
        if self.cuda:
            features, output = self.extractor(input)
        else:
            features, output = self.extractor(input)

        if target_index == None:
            target_index = torch.argmax(output)

        one_hot_enc = self.one_hot_encode(output, target_index)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot_enc.backward(retain_graph=True)

        gradients = self.feature_extractor.gradients[-1] # get gradients of last layer
        # gradients = self.gradients[-1]  # get gradients of last layer

        target = features[-1]
        target = target[0, :]

        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        gcam_map = (weights * target).sum(1, keepdim=True)
        gcam_map = self.cam_activation(gcam_map) # activation
        # reshape
        gcam_map = gcam_map.view((u,v)).cpu().data.numpy()
        # interpolation
        gcam_map = cv2.resize(gcam_map, input.shape[2:])
        # rescale between range [0,1]
        gcam_map = (gcam_map - np.min(gcam_map)) / np.max(gcam_map)

        return gcam_map




if __name__ == '__main__':
    from data.img_processing import load_image, preprocess_image
    from utils.viz import viz_gradcam

    model = models.resnet50(pretrained=True)
    grad_cam = GradCAM(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=False)

    img = load_image('../../datasets/test_images/cat_dog.png')
    input_im = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input_im, target_index)

    viz_gradcam(img, mask, '../../results/test_results/cat_dog_viz.png')

