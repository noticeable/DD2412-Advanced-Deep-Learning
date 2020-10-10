from torchvision import models
import argparse
import torch

def load_model(model_name):
    if model_name == 'resnet18':
        return models.resnet18(pretrained=True)
    elif model_name == 'resnet50':
        return models.resnet50(pretrained=True)
    elif model_name == 'vgg16':
        return models.vgg16(pretrained=True)
    elif model_name == 'alexnet':
        return models.alexnet(pretrained=True)
    elif model_name == 'googlenet':
        return models.googlenet(pretrained=True)
    else:
        print(f'Model {model_name} not yet implemented with grad-cam')
        raise NotImplementedError


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU')
    parser.add_argument('--datapath', type=str, default='../datasets/TODO', help='Path to dataset')
    parser.add_argument('--cam-method', default='grad-cam', options=['grad-cam', 'guided-grad-cam', 'grad-cam++'], help='CAM methos to use')
    parser.add_argument('--model-name', default='resnet18', help='Pretrained model to load')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


if __name__ == '__main__':
    args = get_args()
    model = load_model(args.model_name)
    cam_method = args.cam_method

    # TODO
    if cam_method == 'grad-cam':
        # instantiate grad-cam class with model
        raise NotImplementedError
    elif cam_method == 'guided-grad-cam':
        raise NotImplementedError
    elif cam_method == 'grad-cam++':
        raise NotImplementedError