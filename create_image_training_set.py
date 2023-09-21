# %%
### Import libraries
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets, models
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import splitfolders
# %load_ext tensorboard
import datetime
import time
import zipfile
import csv
# %%
## unzip pytorch image set - only run once
'''
with zipfile.ZipFile('pytorch_images.zip', 'r') as zip_ref:
    zip_ref.extractall()
'''
# %%
os.getcwd()
# %%
pytorch_images = os.path.join(os.getcwd(), 'pytorch_images')
os.remove(os.path.join(pytorch_images, '.DS_Store'))
# %%
## Image classes
class_list = os.listdir(pytorch_images)
## remove all hidden files from class folders
'''
for img_class in class_list:
    os.remove(os.path.join(pytorch_images, img_class, '.DS_Store'))
'''
# %%
## class encoder
encoder = {}
for class_no, class_label in enumerate(class_list):
    encoder[class_label] = class_no
# %%
encoder
# %%
img_class_dict = {}
for img_class in class_list:
    for image in os.listdir(os.path.join(pytorch_images, img_class)):
        img_class_dict[image] = encoder[img_class]
# %%
img_class_dict
# %%
## write the image class mapping to a csv file
img_class_df = pd.DataFrame.from_dict(img_class_dict, orient='index', columns=['class'])
img_class_df['image'] = img_class_df.index
img_class_df = img_class_df.reset_index()
col_order = ['image', 'class']
img_class_df = img_class_df[col_order]
img_class_df.to_csv('pytorch_images_training_data.csv', index=False)
# %%
img_class_df.head()
# %%
## create training and validation splits of image folders and files
splitfolders.ratio('pytorch_images', output='pytorch_images_tv_split', seed=1337, ratio=(0.7, 0.3))
# %%
