"""
Module Name: faiss_indexer
Author: Rupert Coghlan
Date: 21/09/2023
Description: 
"""

import datetime  # Importing datetime module for working with dates and times.
import faiss  # Importing Faiss for similarity search.
import json  # Importing json module for JSON manipulation.
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot for data visualization.
import numpy as np  # Importing numpy for numerical computations.
import os  # Importing os module for operating system-related tasks.
import pandas as pd  # Importing pandas for data manipulation and analysis.
import requests  # Importing requests for making HTTP requests.
from PIL import Image  # Importing Image from the PIL (Pillow) library for image processing.
import time  # Importing time module for working with timestamps.

import torch  # Importing the PyTorch library for deep learning.
import torch.nn as nn  # Importing neural network modules from PyTorch.
import torch.nn.functional as F  # Importing functional operations for PyTorch neural networks.

import torchvision  # Importing the torchvision library for computer vision tasks.
import torchvision.transforms as transforms  # Importing data transformation functions.
from torch.utils.data import Dataset, DataLoader  # Importing data loading utilities from PyTorch.
from torchvision import datasets, models  # Importing pre-trained models and datasets for computer vision.

from torchvision.models.resnet import *  # Importing ResNet models and related modules.
from torchvision.models.resnet import BasicBlock, Bottleneck  # Importing specific ResNet building blocks.

## Leverage the process_img function

def process_img(image):
    """
    Process an input image for use in the modified pretrained resnet50 classifier.

    Parameters
    ----------
    image : str
        The file path of the input image to be processed.

    Returns
    -------
    torch.Tensor
        A processed image represented as a PyTorch tensor with specific dimensions.
    """    
    pil_to_tensor = transforms.ToTensor()
    resize = transforms.Resize((225,225))
    img = Image.open(image).convert('RGB')

    features = pil_to_tensor(img)
    features = resize(features)
    features = torch.unsqueeze(features, dim=0)
    #print(features.shape)
    return features

## Read the embeddings JSON file and convert to a dictionary

with open('Faiss_API/final_embeddings_sep.json', "r") as json_file:
    data_dict = json.load(json_file)

index = faiss.IndexFlatL2(1000)   # build the index, d=size of vectors
# here we assume xb contains a n-by-d numpy matrix of type float32

## Create a flattened array of float32 vectors
embeddings_array = np.array(list(data_dict.values()), dtype='float32')
## Create a maching array of the vector ids (based on the filenames)
embeddings_ids = np.array(list(data_dict.keys()))
## Create the FAISS index by using the add function
index.add(embeddings_array)
print(index.is_trained)

## Created a classifier based on the RESNET50 pretrained model

class ItemFeatureExtractor(torch.nn.Module):
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
        #self.resnet50 = model
        self.resnet50.fc = torch.nn.Linear(2048,1000)

    def forward(self, X):
        return F.softmax(self.resnet50(X))

## Instantiate the model class as the feature extractor with the original parameters
model = ItemFeatureExtractor()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

## Load the final training weights
checkpoint = torch.load('final_weights/weights.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

## Test FAISS by creating a set of query embeddings using the model output and process_image function
query_embeddings = model(process_img('pytorch_images_tv/train/appliances/16.jpg'))

## Flatten the output
query_embeddings = query_embeddings.view(query_embeddings.size(0), -1)

## Convert the output to an float 32 array
query_vector = np.array(list(query_embeddings.tolist()), dtype='float32')

## Test the index search
D, I = index.search(query_vector.reshape(1, -1), 4)



