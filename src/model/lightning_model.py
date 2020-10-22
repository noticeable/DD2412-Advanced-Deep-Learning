''' Lightning Model for finetuning models pretrained on ImageNet.
For default params, run using python model/lightning_model.py --dataset_dir '../datasets/VOC2012
Add any other args through command line as necessary
'''

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from data.datasets import VOC2012, VOC2007, \
    input_transform, target_transform
from utils.logger import create_logger

def load_pretrained_model(model_name, num_classes=20):
    ''' Returns model with correct output layer for finetuning on dataset'''

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    else:
        print(f'Model {model_name} not yet implemented with grad-cam')
        raise NotImplementedError

    return model


class VOCModel(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # init the pretrained model
        self.network = load_pretrained_model(self.params['model_name'], self.params['num_classes'])
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = self.network(x)
        return output

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=self.params['lr'])
        return optimiser

    def get_loss_accuracy(self, pred, target):
        # pred has shape (B, C), target has shape (B, 1) where each value is an integer from 0 to C-1
        loss = self.criterion(pred, target)
        pred = self.softmax(pred)  # TODO check needed
        pred = torch.argmax(pred, axis=1)
        accuracy = (pred == target).sum().item() / len(pred)
        return {'loss': loss, 'acc': accuracy}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x)
        return self.get_loss_accuracy(y_hat, y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x)
        loss_accuracy = self.get_loss_accuracy(y_hat, y)
        loss_accuracy['val_loss'] = loss_accuracy['loss'] # add for checkpointing
        return loss_accuracy

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x)
        return self.get_loss_accuracy(y_hat, y)

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = sum([x['acc'] for x in outputs]) / len(outputs)
        self.log('train_loss', avg_loss)
        self.log('train_acc', avg_accuracy)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = sum([x['acc'] for x in outputs]) / len(outputs)
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_accuracy)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = sum([x['acc'] for x in outputs]) / len(outputs)
        self.log('test_loss', avg_loss)
        self.log('test_acc', avg_accuracy)


    def train_dataloader(self):
        if self.params['segmentation']:
            dataset = VOC2012(f'{self.params["dataset_dir"]}/train', input_transform, target_transform)
        else:
            dataset = VOC2007(self.params["dataset_dir"], split='train', input_transform=input_transform)
        dataloader = torch.utils.data.DataLoader(dataset,
                   num_workers=self.params['n_workers'], batch_size=self.params['batch_size'], shuffle=True)
        return dataloader

    def val_dataloader(self):
        if self.params['segmentation']:
            dataset = VOC2012(f'{self.params["dataset_dir"]}/val', input_transform, target_transform)
        else:
            dataset = VOC2007(self.params["dataset_dir"], split='val', input_transform=input_transform)
        dataloader = torch.utils.data.DataLoader(dataset,
                   num_workers=self.params['n_workers'], batch_size=self.params['batch_size'], shuffle=False)
        return dataloader

    def test_dataloader(self):
        if self.params['segmentation']:
            dataset = VOC2012(f'{self.params["dataset_dir"]}/test', input_transform, target_transform)
        else:
            dataset = VOC2007(self.params["dataset_dir"], split='test', input_transform=input_transform)
        dataloader = torch.utils.data.DataLoader(dataset,
                   num_workers=self.params['n_workers'], batch_size=1, shuffle=False)
        return dataloader


def get_train_params(params):
    trainer_params = params.copy()  # create separate trainer_params as can't pickle model if checkpoint_callback is passed to hparams in model
    experiment_type = 'segmentation' if params['segmentation'] else 'detection'
    model_dir = Path(f"../results/models/{params['model_name']}")

    model_dir = model_dir / experiment_type
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        # filepath=f'{model_save_path}_best_model.pth',
        filepath=model_dir,
        save_top_k=10,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer_params['checkpoint_callback'] = checkpoint_callback
    if not params['disable_logger']:
        experiment_name = 'finetuning_' + params['model_name'] + '_' + experiment_type
        trainer_params['logger'] = create_logger(experiment_name)
    if params['debug']:
        trainer_params.update({'limit_train_batches': 0.01, 'limit_val_batches': 0.02, 'limit_test_batches': 0.03})

    trainer_params = argparse.Namespace(**trainer_params)
    return trainer_params


if __name__ == '__main__':
    # get args and instantiate Trainer
    # TODO @09panesara implement segmentation as well - pass in detection=False to dataset, modify num_classes
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet50', help='model to finetune', choices=['resnet18', 'resent50', 'vgg16', 'alexnet'])
    parser.add_argument('--segmentation', action='store_true', default=False, help='whether to train segmentation. If False, trains detection.')
    parser.add_argument('--num_classes', type=int, default=21, help='number of output classes. 20 (+1 for background) for VOCDetection.')
    parser.add_argument('--dataset_dir', type=str, help='path to directory of data', default='../datasets/VOC2012', choices=['../datasets/VOC/VOC2012', '../datasets/VOC/VOC2007/loader'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=20, help='Max number of epochs to train for')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--disable_logger', action='store_true', default=False, help='whether to disable comet logger')
    parser.add_argument('--gpus', default=None, help='Number of workers for dataloader')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='if true, runs on subset of data.')
    parser.add_argument('--lr', default=0.0001, type=float)


    hparams = vars(parser.parse_args())
    trainer_params = get_train_params(hparams)
    # init lightning module
    model = VOCModel(hparams)
    # trainer
    trainer = Trainer.from_argparse_args(trainer_params)
    trainer.fit(model)
    trainer.test()
    print('Saving logs')
    trainer.save_checkpoint(f"../results/models/{hparams['model_name']}final_model.ckpt")



