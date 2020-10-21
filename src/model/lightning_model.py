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

from data.dataset_voc import VOC12, \
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


class VOCDetectionModel(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # init the pretrained model
        self.network = load_pretrained_model(self.params['model_name'], self.params['num_classes'])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.network(x)
        return output

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=self.params['lr'])
        return optimiser

    def loss_function(self, pred, target):
        # pred has shape (B, C), target has shape (B, 1) where each value is an integer from 0 to C-1
        return self.criterion(pred, target)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x)
        loss = self.loss_function(y_hat, y)
        # TODO add accuracy, add comet_logs
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x)
        loss = self.loss_function(y_hat, y)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x)
        loss = self.loss_function(y_hat, y)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        comet_logs = {'train_loss': avg_loss}
        return {'loss': avg_loss, 'log': comet_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        comet_logs = {'train_loss': avg_loss}
        return {'loss': avg_loss, 'log': comet_logs}

    def train_dataloader(self):
        dataset = VOC12(f'{self.params["dataset_dir"]}/train', input_transform, target_transform)
        dataloader = torch.utils.data.DataLoader(dataset,
                   num_workers=self.params['n_workers'], batch_size=self.params['batch_size'], shuffle=True)
        return dataloader

    def val_dataloader(self):
        # TODO change /train to /val
        dataset = VOC12(f'{self.params["dataset_dir"]}/train', input_transform, target_transform)
        dataloader = torch.utils.data.DataLoader(dataset,
                   num_workers=self.params['n_workers'], batch_size=self.params['batch_size'], shuffle=False)
        return dataloader

    def test_dataloader(self):
        # TODO change /train to /test
        dataset = VOC12(f'{self.params["dataset_dir"]}/val', input_transform, target_transform)
        dataloader = torch.utils.data.DataLoader(dataset,
                   num_workers=self.params['n_workers'], batch_size=1, shuffle=False)
        return dataloader


def get_train_params(params):
    trainer_params = params.copy()  # create separate trainer_params as can't pickle model if checkpoint_callback is passed to hparams in model

    model_dir = Path(f"../results/models/{params['model_name']}")
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
        experiment_name = 'finetuning_' + params['model_name']
        trainer_params['logger'] = create_logger(experiment_name)

    trainer_params = argparse.Namespace(**trainer_params)
    return trainer_params


if __name__ == '__main__':
    # get args and instantiate Trainer
    # TODO @09panesara implement segmentation as well - pass in detection=False to dataset, modify num_classes
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet50', help='model to finetune', choices=['resnet18', 'resent50', 'vgg16', 'alexnet'])
    parser.add_argument('--num_classes', type=int, default=20, help='number of output classes. 20 for VOCDetection.')
    parser.add_argument('--dataset_dir', type=str, help='path to directory of data', default='../datasets/VOC2012')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=20, help='Max number of epochs to train for')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--disable_logger', action='store_true', default=False, help='whether to disable comet logger')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='if true, runs on subset of data.')
    parser.add_argument('--lr', default=0.0001, type=float)


    hparams = vars(parser.parse_args())
    trainer_params = get_train_params(hparams)
    # init lightning module
    model = VOCDetectionModel(hparams)
    # trainer
    trainer = Trainer.from_argparse_args(trainer_params)
    trainer.fit(model)
    trainer.test()
    print('Saving logs')
    trainer.save_checkpoint(f"../results/models/{hparams['model_name']}final_model.ckpt")



