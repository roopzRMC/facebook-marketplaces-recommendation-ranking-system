import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms

## Process image applies the same transformations that were applied during the mmodel training process
def process_img(image):
    """
    Process an input image for deep learning tasks by applying transformations.

    This function takes an input image in PIL format, converts it to a PyTorch tensor,
    resizes it to a specified size, and returns the processed image as a tensor.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image in PIL format.

    Returns
    -------
    torch.Tensor
        A processed image represented as a PyTorch tensor.
    """    
    pil_to_tensor = transforms.ToTensor()
    resize = transforms.Resize((225,225))
    img = image.convert('RGB')
    features = pil_to_tensor(img)
    features = resize(features)
    features = torch.unsqueeze(features, dim=0)
    return features