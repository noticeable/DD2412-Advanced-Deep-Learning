''' Lightning Model for training classifier head on specified layer of (finetuned) resnet50
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


from data.datasets import VOC2007, \
    input_transform
from utils.logger import create_logger
from utils.prepare_models import load_fine_trained


class LinearProbeModel(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # init the pretrained model
        # self.network = self.load_model(self.params['layer_name'], self.params['num_classes'])
        model = load_fine_trained('resnet_50', self.params['checkpoint_path'])
        self.network = self.prepare_model(model.network, self.params['layer_name'], self.params['num_classes'])
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.save_hyperparameters()

    def freeze_layers(self, model, layer_up_to):
        # freeze up to layer lay_up_to (not incl)
        # 10 children layers - conv, bn, relu, maxpool, layer1, layer2, layer2, layer4, avgpool, fc
        child_counter = 0
        for child in model.children():
            if child_counter < layer_up_to:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print("child ", child_counter, " was not frozen")
            child_counter += 1
        return model

    def prepare_model(self, model, layer_name, num_classes):
        # attaches classifier head after layer_name
        # model = models.resnet50(pretrained=True)
        if layer_name == 4: # last layer
            model.fc = nn.Linear(2048, num_classes)
            model.fc.requires_grad = True
            self.freeze_layers(model, 9) # freeze all but fc
            return model

        avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        flatten = nn.Flatten()
        avgpool.requires_grad = True

        model = self.freeze_layers(model, 10) # freeze all layers

        if layer_name == 1:
            layers = list(model.children())[:-5]
            layers.append(avgpool)
            fc = nn.Linear(256, num_classes)
            fc.requires_grad = True
            layers.append(flatten)
            layers.append(fc)
            model = nn.Sequential(*layers)

        elif layer_name == 2:
            layers = list(model.children())[:-4]
            layers.append(avgpool)
            fc = nn.Linear(512, num_classes)
            fc.requires_grad = True
            layers.append(flatten)
            layers.append(fc)
            model = nn.Sequential(*layers)
        elif layer_name == 3:
            layers = list(model.children())[:-3]
            layers.append(avgpool)
            fc = nn.Linear(1024, num_classes)
            fc.requires_grad = True
            layers.append(flatten)
            layers.append(fc)
            model = nn.Sequential(*layers)
        else:
            print('layer', layer_name, 'not implemented yet')
            raise NotImplementedError
        return model

    def forward(self, x):
        output = self.network(x)
        return output

    def configure_optimizers(self):
        # optimiser = torch.optim.Adam(self.parameters(), lr=self.params['lr'])

        optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.params['lr'])
        self.log('lr', self.params['lr'])
        return optimiser

    def get_loss_accuracy(self, pred, target):
        # pred has shape (B, C), target has shape (B, 1) where each value is an integer from 0 to C-1
        loss = self.criterion(pred, target)
        pred = self.softmax(pred)
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
        dataset = VOC2007(self.params["dataset_dir"], split='train', input_transform=input_transform)
        dataloader = torch.utils.data.DataLoader(dataset,
                   num_workers=self.params['n_workers'], batch_size=self.params['batch_size'], shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataset = VOC2007(self.params["dataset_dir"], split='val', input_transform=input_transform)
        dataloader = torch.utils.data.DataLoader(dataset,
                   num_workers=self.params['n_workers'], batch_size=self.params['batch_size'], shuffle=False)
        return dataloader

    def test_dataloader(self):
        dataset = VOC2007(self.params["dataset_dir"], split='test', input_transform=input_transform)
        dataloader = torch.utils.data.DataLoader(dataset,
                   num_workers=self.params['n_workers'], batch_size=1, shuffle=False)
        return dataloader


def get_train_params(params):
    trainer_params = params.copy()  # create separate trainer_params as can't pickle model if checkpoint_callback is passed to hparams in model
    model_dir = Path(f"../results/linear_probing/resnet_50/layer_{hparams['layer_name']}")
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=model_dir,
        save_top_k=10,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer_params['checkpoint_callback'] = checkpoint_callback
    if not params['disable_logger']:
        experiment_name = 'finetuning_resnet_' + str(params['layer_name'])
        trainer_params['logger'] = create_logger(experiment_name)
    if params['debug']:
        trainer_params.update({'limit_train_batches': 0.01, 'limit_val_batches': 0.02, 'limit_test_batches': 0.03})

    trainer_params = argparse.Namespace(**trainer_params)
    return trainer_params


if __name__ == '__main__':
    # get args and instantiate Trainer
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, help='name of experiment', default='test_experiment')
    parser.add_argument('--layer_name', type=int, default=1, choices=[1,2,3,4])
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
    parser.add_argument('--checkpoint_path', default='../model.ckpt', type=str)
    parser.add_argument('--test_only', action='store_true', default=False)


    hparams = vars(parser.parse_args())
    trainer_params = get_train_params(hparams)



    if not(hparams['test_only']):
        # trainer
        model = LinearProbeModel(hparams)       # init lightning module
        trainer = Trainer.from_argparse_args(trainer_params)
        trainer.fit(model)
        trainer.test()
        print('Saving logs')
        trainer.save_checkpoint(f"../results/linear_probing/resnet_50/layer_{hparams['layer_name']}/final_model.ckpt")
    else:
        # test model
        model_test = LinearProbeModel.load_from_checkpoint(hparams['checkpoint_path'])
        trainer = Trainer.from_argparse_args(trainer_params)
        trainer.test(model_test)




