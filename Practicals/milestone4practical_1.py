# %%
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import glob
import csv

# %%
os.getcwd()
# %%
def get_class_names():
    assert os.path.exists('CustomDataset'), 'CustomDataset directory cannot be found'
    return [f for f in os.listdir('CustomDataset') if not f.startswith('.')]

# %%
class_names = get_class_names()
# %%
## Encode and decode the images
encoder = {instrument: i for i, instrument in enumerate(class_names)}
decoder = {i: instrument for i, instrument in enumerate(class_names)}
# %%
print(encoder)
print(decoder)
# %%

# %%
## Create image to class dictionary
image_class_dict = {}
for img_class in class_names:
    img_path = os.path.join('CustomDataset', img_class)
    for image in os.listdir(img_path):
        image_class_dict[image] = img_class
# %%
image_class_dict
image_class_dict.pop('.DS_Store')

# %%
## Create the training data CSV file

csv_columns = ['image', 'image_class']
csv_file = 'training_data.csv'
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key in image_class_dict.keys():
            csvfile.write("%s,%s\n"%(key,image_class_dict[key]))
except IOError:
    print("I/O error")    


# %%
## ingest the annotation CSV
image_annotation = pd.read_csv('training_data.csv')

# %%
image_annotation

# %%
class InstrumentDataset(Dataset):
    def __init__(self):
        self.instruments = get_class_names()
        self.encoder = encoder
        self.decoder = decoder
        self.all_imgs = []

        for instrument in self.instruments:
            for image in os.listdir('CustomDataset/'+instrument):
                if not image.startswith('.'):
                    self.all_imgs.append(image)
                    continue
        
    def __len__(self):
        return len(self.all_imgs)
    
    def __getitem__(self, index):
        img_cl_map = []
        for instrument in class_names:
            for image in os.listdir('CustomDataset/'+instrument):
                if not image.startswith('.'):
                    img_cl_map.append((image, instrument))
        img = Image.open(os.path.join('CustomDataset/', img_cl_map[index][1], img_cl_map[index][0])) 
        convert_tensor = transforms.ToTensor()
        img = convert_tensor(img)
        label = img_cl_map[index][1]
        return img, label
# %%
## Instantiate the dataset class
idtest = InstrumentDataset()
# %%
## Check the get item functionality
idtest[4]
# %%
## Check the len functionality
len(idtest)
# %%
