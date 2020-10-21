import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU')
    parser.add_argument('--data-dir', type=str, default='../datasets/TODO', help='Path to dataset dir to test on')
    parser.add_argument('--cam-method', default='grad-cam', options=['grad-cam', 'guided-grad-cam', 'grad-cam++'], help='CAM methos to use')
    parser.add_argument('--model-path', help='Path to model to load')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def visualise_heatmap(cam_method):
    if cam_method == 'grad-cam':
        # instantiate grad-cam class with model
        raise NotImplementedError
    elif cam_method == 'guided-grad-cam':
        raise NotImplementedError
    elif cam_method == 'grad-cam++':
        raise NotImplementedError


if __name__ == '__main__':
    args = get_args()
    cam_method = args.cam_method
    visualise_heatmap(cam_method)

