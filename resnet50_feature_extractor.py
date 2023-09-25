"""
Module Name: resnet50_feature_extractor
Author: Rupert Coghlan
Date: 21/09/2023
Description: This script instantiates a pretrained resnet 50 and trains on a custom datset of 13 classes. The weights are saved in model evaluation
"""

# Import datetime and time for date and time operations
import datetime
import time

# Import json for JSON manipulation
import json

# Import os for file-related activities and folder navigation
import os

# Import the requests library for HTTP requests
import requests

# Import warnings for managing warnings
import warnings
warnings.filterwarnings('ignore')

# Import NumPy for numerical operations
import numpy as np

# Import Pandas for data manipulation
import pandas as pd

# Import the Python Imaging Library (PIL) for image processing
from PIL import Image

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import specific components from torchvision.models.resnet
from torchvision.models.resnet import BasicBlock, Bottleneck

# Import PyTorch Dataset and DataLoader for data handling
from torch.utils.data import Dataset, DataLoader

# Import PyTorch torchvision.transforms for image transformations during training
import torchvision.transforms as transforms

# Import PyTorch torchvision.datasets and models
from torchvision import datasets, models

# Import torchvision for additional image-related functionality
import torchvision

# Import SummaryWriter from torch.utils.tensorboard for TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

# Clear GPU cache using torch.cuda.empty_cache()
torch.cuda.empty_cache()

# Determine the device (GPU or CPU) available for computation - ensure that cuda is displayed
device = torch.device("this session is using cuda as a torch device") if torch.cuda.is_available() else torch.device("this session is using a cpu as a torch device")
print(device)

## Creating a train dataset class
class ItemsTrainDataSet(Dataset):
    """
    A custom dataset class for image training data.
    
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
        #features = self.rgbify(features)

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
    A custom dataset class for image validation data.
    
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
        #self.rgbify = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)!=1 else x)

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
        #features = self.rgbify(features)

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

class ItemFeatureExtractor(torch.nn.Module):
    """
    A custom nn.Module class housing the feature extractor which is
    based on a gpu-derived pretrained resnet50 from NVIDA torchhub
    
    Attributes
    ----------
    None
    
    
    Methods
    -------
    __init__():
        Initialises the classifier and loads the model from the torchhub
        unless it is able to detect a cached instance.
        Replaces the final layer with a 1000 way neuron for extraction to FAISS
    
    forward():
        Initiates the forward pass
        
    """        
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048,1000)
  
    def forward(self, X):
        return F.softmax(self.resnet50(X))

def train(model,traindataloader, valdataloader, epochs):
    """
    Train the feature extractor with training and validation datasets.

    Parameters
    ----------
    model : torch.nn.Module
        The feature extractor to be trained.
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model_path = str('final_model')   
    os.makedirs(model_path)
    os.makedirs(os.path.join(model_path, 'weights'))
    batch_idx = 0
    
    for epoch in range(epochs):
        training_loss = 0.0
        validation_loss = 0.0
        model.to(device)
        model.train()
        tr_num_correct = 0
        tr_num_examples = 0
        for inputs, labels in traindataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model(inputs)
            loss = torch.nn.CrossEntropyLoss()
            loss = loss(predictions, labels)
            loss.backward()
            optimiser.step()
            batch_idx += 1
            training_loss += loss.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(predictions, dim=1), dim=1)[1], labels)
            tr_num_correct += torch.sum(correct).item()
            tr_num_examples += correct.shape[0]
        training_loss /= len(traindataloader.dataset)

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
        perf_dict = {}
        perf_dict[epoch] = {'training_loss': training_loss,
                            'val_loss': validation_loss,
                            'training_accuracy': tr_num_correct / tr_num_examples,
                            'val_accuracy': val_num_correct / val_num_examples}
                            
                            
        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, train_accuracy = {:.2f},val_accuracy = {:.2f} '.format(epoch, training_loss, validation_loss, tr_num_correct / tr_num_examples,
                                                                                                                             val_num_correct / val_num_examples))
    model_save_dir = str(os.path.join(model_path, 'weights', 'weights.pt'))
    full_path = str('/home/ubuntu/facebook-marketplaces-recommendation-ranking-system/Practicals')
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimiser.state_dict()},
    str(os.path.join(full_path, model_save_dir)))

## Instantiate the feature extractor
Extractor = ItemFeatureExtractor()

## Create the train and validation loaders
## Pass the loaders, extractor and desired number of epochs to the train function define above
train_loader = DataLoader(dataset = traindataset, batch_size=16)
val_loader = DataLoader(dataset = valdataset, batch_size=16)
train(Extractor, traindataloader= train_loader, valdataloader= val_loader, epochs=1)