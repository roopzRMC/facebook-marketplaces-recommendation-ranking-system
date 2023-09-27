"""
Module Name: resnet50_classifier
Author: Rupert Coghlan
Date: 21/09/2023
Description: This script instantiates a pretrained resnet 50 and trains on a custom datset of 13 classes. The weights are saved in model evaluation
"""

# Date and time operations
import datetime
import time

# JSON manipulation
import json

# File-related activities and folder navigation
import os

# HTTP requests
import requests

# Managing warnings
import warnings

# Numerical operations
import numpy as np

# Data manipulation
import pandas as pd

# Pytorch functionality
import torch

# Image-related functionality
import torchvision

# Image transformations during training
import torchvision.transforms as transforms

# Image processing
from PIL import Image

# Pytorch Neural network modules
from torch.nn import nn
from torch.nn.functional import F

# Dataset and DataLoader for data handling
from torch.utils.data import Dataset, DataLoader

# TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

# Computer vision models
from torchvision import datasets, models
from torchvision.models.resnet import BasicBlock, Bottle

# Determine the device (GPU or CPU) available for computation - ensure that cuda is displayed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

class ItemsTrainDataSet(Dataset):
    """
    A custom dataset class for image training data which creates the model training image dataset.
    
    This class inherits from the PyTorch Dataset class and will 
    load and preprocess image training data.

    Attributes
    ----------
    None

    Methods
    -------
    __init__():
        Initializes a new ItemsTrainDataSet instance.
    _load_examples():
        Loads and prepares a list of image examples.
    __getitem__(idx):
        Retrieves an image and its associated class label by index.
    __len__():
        Returns the total number of examples in the dataset.
    """
    def __init__(self):
        """
        Initialize a new ItemsTrainDataSet instance.

        This constructor sets up the necessary transformations for image preprocessing.
        """        
        super().__init__()
        self.examples = self._load_examples()
        self.pil_to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((225,225))
        

    def _load_examples(self):
        """
        Load and prepare a list of image examples.

        This method scans the training directory, assigns class labels,
        and collects a list of image file paths.

        Returns
        -------
        list
            A list of tuples containing image file paths and their class labels.
        """        
        class_names = os.listdir('pytorch_images_tv/train')
        class_encoder = {class_name: idx for idx, class_name in enumerate(class_names)}
        class_decoder = {idx: class_name for idx, class_name in enumerate(class_names)}

        examples_list = []
        for cl_name in class_names:
            example_fp = os.listdir(os.path.join('pytorch_images_tv/train',cl_name))
            example_fp = [os.path.join('pytorch_images_tv/train', cl_name, img_name ) for img_name in example_fp]
            example = [(img_name, class_encoder[cl_name]) for img_name in example_fp]
            examples_list.extend(example)

        return examples_list

    def __getitem__(self, idx):
        """
        Retrieve an image and its associated class label by index.

        Parameters
        ----------
        idx : int
            The index of the item to retrieve.

        Returns
        -------
        tuple
            A tuple containing the preprocessed image and its class label.
        """        
        img_fp, img_class = self.examples[idx]
        img = Image.open(img_fp)

        features = self.pil_to_tensor(img)
        features = self.resize(features)

        return features, img_class

    def __len__(self):
        """
        Return the total number of examples in the dataset.

        Returns
        -------
        int
            The total number of examples in the dataset.
        """        
        return len(self.examples)

## Creates a validation dataset class
class ItemsValDataSet(Dataset):
    """
    A custom dataset class for image validation data which creates the image validation set.
    
    This class inherits from the PyTorch Dataset class and is designed
    for loading and preprocessing image validation data.

    Attributes
    ----------
    None

    Methods
    -------
    __init__():
        Initializes a new ItemsValDataSet instance.
    _load_examples():
        Loads and prepares a list of image examples.
    __getitem__(idx):
        Retrieves an image and its corresponding class label by index.
    __len__():
        Returns the total number of examples in the dataset.
    """
    def __init__(self):
        """
        Initialize a new ItemsValDataSet instance.

        This constructor sets up the necessary transformations for image preprocessing.
        """
        super().__init__()
        self.examples = self._load_examples()
        self.pil_to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((225,225))
        

    def _load_examples(self):
        """
        Load and prepare a list of image examples.

        This method scans the validation directory, assigns class labels,
        and collects a list of image file paths.

        Returns
        -------
        list
            A list of tuples containing image file paths and their class labels.
        """        
        class_names = os.listdir('pytorch_images_tv/val')
        class_encoder = {class_name: idx for idx, class_name in enumerate(class_names)}
        class_decoder = {idx: class_name for idx, class_name in enumerate(class_names)}
        examples_list = []

        for cl_name in class_names:
            example_fp = os.listdir(os.path.join('pytorch_images_tv/val',cl_name))
            example_fp = [os.path.join('pytorch_images_tv/val', cl_name, img_name ) for img_name in example_fp]
            example = [(img_name, class_encoder[cl_name]) for img_name in example_fp]
            examples_list.extend(example)

        return examples_list

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding class label by index.

        Parameters
        ----------
        idx : int
            The index of the item to retrieve.

        Returns
        -------
        tuple
            A tuple containing the preprocessed image and its class label.
        """        
        img_fp, img_class = self.examples[idx]
        img = Image.open(img_fp)

        features = self.pil_to_tensor(img)
        features = self.resize(features)

        return features, img_class

    def __len__(self):
        """
        Return the total number of examples in the dataset.

        Returns
        -------
        int
            The total number of examples in the dataset.
        """        
        return len(self.examples)


# Create the traindataset object 
traindataset = ItemsTrainDataSet()

# Create the validation dataset object
valdataset = ItemsValDataSet()

class ItemClassifier(torch.nn.Module):
    """
    A custom nn.Module class housing the classifier which is
    based on a gpu-derived pretrained resnet50 from NVIDA torchhub
    
    Attributes
    ----------
    None
    
    
    Methods
    -------
    __init__():
        Initialises the classifier and loads the model from the torchhub
        unless it is able to detect a cached instance.
        Replaces the final layer with a 13 class output
    
    forward():
        Initiates the forward pass
        
    """    
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048,13)

    def forward(self, X):
        return F.softmax(self.resnet50(X))

def train(model,traindataloader, valdataloader, epochs):
    """
    Train the modified pretrained resnet50 classifier.
    
    The model weights are written after each epoch to model_evaluation in
    the current working directory
    
    In addition at the end of each epoch:
        Validation Accuracy
        Validation Loss
        Training Accuracy
        Training Loss
    are written to the logs file for the tensorboard to visualise
    performance

    Parameters
    ----------
    model : torch.nn.Module
        The deep learning model to be trained.
    traindataloader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    valdataloader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    epochs : int
        The number of training epochs.

    Returns
    -------
    None
    """    
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    model_path = str(os.path.join('model_evaluation', time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(model_path)
    os.makedirs(os.path.join(model_path, 'weights'))

    global_step = 0

    for epoch in range(epochs):
        training_loss = 0.0
        validation_loss = 0.0
        model.to(device)
        model.train()
        tr_num_correct = 0
        tr_num_examples = 0
        epoch_combo = 'epoch' + str(epoch)
        os.makedirs(os.path.join(model_path, 'weights', epoch_combo))
        for inputs, labels in traindataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model(inputs)
            loss = torch.nn.CrossEntropyLoss()
            loss = loss(predictions, labels)
            loss.backward()
            optimiser.step()
            model_save_dir = str(os.path.join(model_path, 'weights', epoch_combo, 'weights.pt'))
            full_path = str(os.getcwd())
            torch.save({'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimiser.state_dict()},
                  str(os.path.join(full_path, model_save_dir)))

            optimiser.zero_grad()
            training_loss += loss.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(predictions, dim=1), dim=1)[1], labels)
            tr_num_correct += torch.sum(correct).item()
            tr_num_examples += correct.shape[0]
        training_loss /= len(traindataloader.dataset)
        training_accuracy = tr_num_correct / tr_num_examples

        model.eval()
        val_num_correct = 0
        val_num_examples = 0
        for inputs, labels in valdataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model(inputs)
            loss = torch.nn.CrossEntropyLoss()
            loss = loss(predictions, labels)
            validation_loss += loss.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(predictions, dim =1), dim=1)[1], labels)
            val_num_correct += torch.sum(correct).item()
            val_num_examples += correct.shape[0]
        validation_loss /= len(valdataloader.dataset)
        validation_accuracy = val_num_correct / val_num_examples

        perf_dict = {}
        perf_dict[epoch] = {'training_loss': training_loss,
                            'val_loss': validation_loss,
                            'training_accuracy': tr_num_correct / tr_num_examples,
                            'val_accuracy': val_num_correct / val_num_examples}


        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, train_accuracy = {:.2f},val_accuracy = {:.2f}'.format(
            epoch,
            training_loss,
            validation_loss, 
            tr_num_correct / tr_num_examples, 
            val_num_correct / val_num_examples))
        global_step += 1

## Create the classifier object from the ItemClassifier class
classifier = ItemClassifier()

## define the layers to unfreeze and then retrain
layers_to_unfreeze = ['layers.2', 'layers.3']

for name, param in classifier.resnet50.named_parameters():
    for layer_name in layers_to_unfreeze:
        if layer_name in name:
            param.requires_grad = True
            break

## Create the train and validation loaders
## Pass the loaders, classifier and desired number of epochs to the train function define above
train_loader = DataLoader(dataset = traindataset, batch_size=16)
val_loader = DataLoader(dataset = valdataset, batch_size=16)
train(classifier, traindataloader= train_loader, valdataloader= val_loader, epochs=150)