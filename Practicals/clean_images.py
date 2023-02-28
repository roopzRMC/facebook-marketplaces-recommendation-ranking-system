# %%
## Load the libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import plotly
from tqdm import tqdm  

# %%
## Check the images and the location
image_dir = 'images'
os.makedirs('clean_images')
clean_image_dir = 'clean_images'
print(f'there are {len(os.listdir(image_dir))} images in the directory \n')
# %%
## Define a resize function
def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

# %%
'''
for each element of the list created through os.listdir of the image
convert to grayscale as the images are a mix of both RGB and grayscale - these need to be unified
The resize function is then applied to each element in the for loop
The new image is saved to the clean images dir
As there are over 12k images, we have used tqdm to provide an ETA and status bar in the console

'''

for i in tqdm(range(len(os.listdir(image_dir)))):
    image = Image.open(os.path.join('images',os.listdir(image_dir)[i])).convert('L')
    clean_image = resize_image(300, image)
    clean_image.save(os.path.join('clean_images',os.listdir(image_dir)[i]))

