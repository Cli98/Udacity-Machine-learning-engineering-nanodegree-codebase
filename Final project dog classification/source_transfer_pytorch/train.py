from __future__ import print_function # future proof
import sys
import argparse
import json
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torchvision.models as models

import torchvision.transforms as transforms
from torchvision import datasets
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

def _get_data_loader(batch_size, base_dir):
    # base_dir -> data_dir
    print("Get data loader.")
    training_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "valid")
    test_dir = os.path.join(base_dir, "test")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(training_dir,
                                         transforms.Compose([transforms.Resize(256),
                                                             transforms.RandomResizedCrop(224),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             normalize]))
    val_dataset = datasets.ImageFolder(val_dir,
                                   transforms.Compose([transforms.Resize(256),
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      normalize]))
    test_dataset = datasets.ImageFolder(test_dir,
                                   transforms.Compose([transforms.Resize(256),
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      normalize]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size,shuffle = False, pin_memory = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size,shuffle = False, pin_memory = True)
    loaders_scratch = {"train":train_loader, "valid":val_loader, "test":test_loader}
    return loaders_scratch

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        train_item, val_item = 0, 0
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        train_item = batch_idx + 1
        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            pred = model(data)
            loss = criterion(pred, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        val_item = batch_idx + 1
        train_loss /= train_item
        valid_loss /= val_item
        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)
    # return trained model
    return model

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job

    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)
    # move tensors to GPU if CUDA is available
    use_cuda = torch.cuda.is_available()
    # Load the training data.
    data_loader = _get_data_loader(args.batch_size, args.data_dir)
    # instantiate the CNN
    model_transfer = models.vgg19(pretrained=True)
    model_transfer.classifier[-1] = nn.Linear(4096, 133, bias=True)  
    if use_cuda:
        model_transfer = model_transfer.cuda()
    criterion_scratch = nn.CrossEntropyLoss()
    optimizer_scratch = optim.SGD(model_transfer.parameters(), lr=args.lr)
    # train the model
    model_path = os.path.join(args.model_dir, 'model_scratch.pth')
    model_scratch = train(args.epochs, data_loader, model_transfer, optimizer_scratch,
                          criterion_scratch, use_cuda, model_path)