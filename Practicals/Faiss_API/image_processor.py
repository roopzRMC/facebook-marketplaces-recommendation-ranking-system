import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms

###

## Process image applies the same transformations that were applied during the mmodel training process
def process_img(image):
    pil_to_tensor = transforms.ToTensor()
    resize = transforms.Resize((225,225))
    img = image.convert('RGB')
    features = pil_to_tensor(img)
    features = resize(features)
    features = torch.unsqueeze(features, dim=0)
    return features