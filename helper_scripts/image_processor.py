"""
Module Name: image_processor
Author: Rupert Coghlan
Date: 21/09/2023
Description: This module creates an embeddings JSON through processing each image via the feature extractor
"""


import datetime  # Importing datetime module for working with dates and times.
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

# Empty the gpu cache
torch.cuda.empty_cache()

# Determine the device (GPU or CPU) available for computation - ensure that cuda is displayed
device = torch.device("this session is using cuda as a torch device") if torch.cuda.is_available() else torch.device("this session is using a cpu as a torch device")
print(device)

## Creating a train dataset class
class ItemsTrainDataSet(Dataset):
    def __init__(self):
        super().__init__()
        self.examples = self._load_examples()
        self.pil_to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((225,225))
        #self.rgbify = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)

    def _load_examples(self):
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
        img_fp, img_class = self.examples[idx]
        img = Image.open(img_fp)

        features = self.pil_to_tensor(img)
        features = self.resize(features)
        #features = self.rgbify(features)

        return features, img_class

    def __len__(self):
        return len(self.examples)

traindataset = ItemsTrainDataSet()
train_loader = DataLoader(dataset = traindataset, batch_size=64)

def process_img(image):
    pil_to_tensor = transforms.ToTensor()
    resize = transforms.Resize((225,225))
    img = Image.open(image).convert('RGB')

    features = pil_to_tensor(img)
    features = resize(features)
    features = torch.unsqueeze(features, dim=0)
    #print(features.shape)
    return features

r

class ItemFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        #self.resnet50 = model
        self.resnet50.fc = torch.nn.Linear(2048,1000)

    def forward(self, X):
        return F.softmax(self.resnet50(X))

## Instantiate a 1000 neuron feature extracto
model = ItemFeatureExtractor()

## Instantiate the optimiser used
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

## Load the feature extractor weights
checkpoint = torch.load('final_weights/weights.pt')

## Load the model state
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

## loop through images to get embeddings

class_list = os.listdir('pytorch_images_tv/train')

#####PROCESS EACH TRAINING IMAGE THROUGH THE MODEL AND WRITE EMBEDDINGS TO A DICTIONARY######

images_dir = 'pytorch_images_tv/train'
embeddings_dict = {}

for cat in class_list:
  for i in os.listdir(os.path.join(images_dir, cat)):
    image_fp = os.path.join(images_dir, cat, i)
    batch_embeddings = model(process_img(os.path.join(images_dir, cat, i)))
    batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)

    for emb, file_name in zip(batch_embeddings, image_fp):
        # Get the image ID from the file name (assuming it is the filename without extension)
        image_id = image_fp.split(".")[0]

        # Create a dictionary for each embedding entry
        embeddings_dict[image_id] = emb.tolist()

#### Write the embeddings dictionary to a JSON file
# Specify the path to your JSON file
json_file_path = "final_embeddings_sep.json"

# Open the file in write mode
with open(json_file_path, 'w') as file:
    # Write the dictionary to the JSON file
    json.dump(embeddings_dict, file)

## Write the json embeddings to a final
with open('final_embeddings_sep.json', "r") as json_file:
  data_dict = json.load(json_file)
