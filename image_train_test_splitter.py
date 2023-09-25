"""
Module Name: image_train_test_splitter
Author: Rupert Coghlan
Date: 21/09/2023
Description: This module downloads pytorch from google blob stoage allowing the user to split the dataset in to train and validation sets by specifying a split ratio
"""


### Import libraries
import csv  # Importing csv module for working with CSV files.
import datetime  # Importing datetime module for working with dates and times.
import google.cloud.storage  # Importing Google Cloud Storage for cloud storage operations.
import json  # Importing json module for JSON manipulation.
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot for data visualization.
import numpy as np  # Importing numpy for numerical computations.
import os  # Importing os module for operating system-related tasks.
import pandas as pd  # Importing pandas for data manipulation and analysis.
import requests  # Importing requests for making HTTP requests.
import shutil  # Importing shutil module for file operations.
import time  # Importing time module for working with timestamps.
import zipfile  # Importing zipfile module for working with ZIP archives.

import random  # Importing random module for generating random values.
import splitfolders  # Importing splitfolders for splitting datasets.
import torch  # Importing the PyTorch library for deep learning.
import torch.nn.functional as F  # Importing functional operations for PyTorch neural networks.
import torchvision  # Importing the torchvision library for computer vision tasks.
import torchvision.transforms as transforms  # Importing data transformation functions.
from PIL import Image  # Importing Image from the PIL (Pillow) library for image processing.
from torch.utils.data import Dataset, DataLoader  # Importing data loading utilities from PyTorch.
from torch.utils.tensorboard import SummaryWriter  # Importing TensorBoard for visualization.
from torchvision import datasets, models  # Importing pre-trained models and datasets for computer vision.

from tqdm import tqdm  # Importing tqdm for creating progress bars.

# Initialise a client
storage_client = storage.Client("aicore-study")
# Create a bucket object for our bucket
bucket = storage_client.get_bucket('pytorch_training_images_13_class')
# Create a blob object from the filepath
blob = bucket.blob("pytorch_images.zip")
# Download the file to a destination
blob.download_to_filename("pytorch_images.zip")

with zipfile.ZipFile('pytorch_images.zip', 'r') as zip_ref:
    zip_ref.extractall()


## Set the pytorch images filepath
pytorch_images = os.path.join(os.getcwd(), 'pytorch_images')

## Remove residual OSX hidden files
os.remove(os.path.join(pytorch_images, '.DS_Store'))

# Create image classes from folders within filepath
class_list = os.listdir(pytorch_images)

## remove all hidden files from class folders
for img_class in class_list:
    os.remove(os.path.join(pytorch_images, img_class, '.DS_Store'))

## Create the class encoder
encoder = {}
for class_no, class_label in enumerate(class_list):
    encoder[class_label] = class_no

## Remove non rgb images from the image directories / subdirectories
for img_class in class_list:
    for image in os.listdir(os.path.join(pytorch_images, img_class)):
        if not image.lower().endswith(('.jpg', '.jpeg')):
            os.remove(os.path.join(os.getcwd(),pytorch_images,img_class,image))

## Convert non RGB images to RGB standard
for img_class in class_list:
    for image in os.listdir(os.path.join(pytorch_images, img_class)):
        image_fp = os.path.join(pytorch_images, img_class, str(image))
        image_obj = Image.open(os.path.join(pytorch_images, img_class, image)).convert('RGB').save(image_fp)

## Create the image class dict with image file and class as key value pairs
img_class_dict = {}
for img_class in class_list:
    for image in os.listdir(os.path.join(os.getcwd(), pytorch_images, img_class)):
        img_class_dict[image] = encoder[img_class]

## write the image class mapping to a csv file
img_class_df = pd.DataFrame.from_dict(img_class_dict, orient='index', columns=['class'])
img_class_df['image'] = img_class_df.index
img_class_df = img_class_df.reset_index()
col_order = ['image', 'class']
img_class_df = img_class_df[col_order]
img_class_df.to_csv('pytorch_images_training_data.csv', index=False)

## Test the output of the resulting dataframe
img_class_df.head()

## Using split folders, create a new image directory with train and validation sub directories 
## Split is 70% and 30% across training and validation folders 
splitfolders.ratio('pytorch_images', output='pytorch_images_tv_split_2', seed=42, ratio=(0.7,0.3))

## randomly pick 50 images from the master dir and push to another copy of the dir
os.mkdir('pytorch_images_lite')
source_dir = 'pytorch_images/'
destination_dir = 'pytorch_images_lite'

## For initial model testing randomly pick files from the 13 categories to a new folder structure
for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)
    if os.path.isdir(folder_path):  # Only process subdirectories
        files = os.listdir(folder_path)
        if len(files) > 100:
            # Randomly sample 50 files
            sampled_files = random.sample(files, 100)

            # Create a new folder in the destination directory
            new_folder_path = os.path.join(destination_dir, folder)
            os.makedirs(new_folder_path, exist_ok=True)

            # Copy the sampled files to the new folder
            for file_name in sampled_files:
                source_file_path = os.path.join(folder_path, file_name)
                destination_file_path = os.path.join(new_folder_path, file_name)
                shutil.copy2(source_file_path, destination_file_path)


## Use split folders to take the samples category splits and spread across a new set of train and validation subdirectories
splitfolders.ratio('pytorch_images_lite', output='pytorch_images_lite_split', seed=42, ratio=(0.7,0.3))