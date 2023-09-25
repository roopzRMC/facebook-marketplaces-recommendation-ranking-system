import fastapi  # Importing the FastAPI framework for building web applications.
import faiss  # Importing Faiss for similarity search.
import json  # Importing json module for JSON manipulation.
import numpy as np  # Importing numpy for numerical computations.
import pickle  # Importing pickle for object serialization.
import torch  # Importing the PyTorch library for deep learning.
import torch.nn as nn  # Importing neural network modules from PyTorch.
import torch.nn.functional as F  # Importing functional operations for PyTorch neural networks.
import torchvision  # Importing the torchvision library for computer vision tasks.
from PIL import Image  # Importing Image from the PIL (Pillow) library for image processing.
from fastapi import File, UploadFile, Form  # Importing FastAPI modules for file uploads and forms.
from fastapi.responses import JSONResponse  # Importing FastAPI's JSONResponse for handling JSON responses.
from pydantic import BaseModel  # Importing Pydantic's BaseModel for defining data models.
import image_processor  # Importing a custom module named "image_processor."

## Load the feature extractor

class ItemFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        #self.resnet50 = model
        self.resnet50.fc = torch.nn.Linear(2048,1000)

    def forward(self, X):
        return F.softmax(self.resnet50(X))

try:
    model = ItemFeatureExtractor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    checkpoint = torch.load('final_model/weights.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")


## Load the FAISS index
try:
    index = faiss.IndexFlatL2(1000)
    with open('index.pickle', 'rb') as file:
        index = pickle.load(file)
    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")


app = fastapi.FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

@app.post('/predict/feature_embedding')


## Function to generate embeddings from supplied image
def predict_image(image: UploadFile = File(...)):
    """
    Process and predict embeddings for an uploaded image.

    This function takes an uploaded image file, processes it to obtain embeddings
    using a pre-trained model, and returns the embeddings as a NumPy array.

    Parameters
    ----------
    image : UploadFile
        An uploaded image file to be processed and used for prediction.

    Returns
    -------
    np.ndarray
        An array of embeddings representing the image.
    """
    pil_image = Image.open(image.file)
    features = image_processor.process_img(pil_image)
    embeddings = model(features)
    embeddings = embeddings.view(embeddings.size(0), -1)
    embeddings = np.array(list(embeddings.tolist()), dtype='float64')
    ## float 32 is not serialisable by JSON but float64 is hence the conversion at the end!

    return JSONResponse(content={
    "features": list(embeddings[0])
    
        })

@app.post('/predict/similar_images')


def predict_combined(image: UploadFile = File(...)):
    """
    Predict similar images for an uploaded image based on pre-computed embeddings.

    This function takes an uploaded image file, processes it to obtain embeddings
    using a pre-trained model, and finds similar images based on the computed embeddings.

    Parameters
    ----------
    image : UploadFile
        An uploaded image file to be used for prediction.

    Returns
    -------
    list
        A list of filenames representing similar images to the input image.
    """    
    with open('final_embeddings.json', "r") as json_file:
        data_dict = json.load(json_file)
    ## Create a flattened array of float32 vectors
    embeddings_array = np.array(list(data_dict.values()), dtype='float32')
    ## Create a maching array of the vector ids (based on the filenames)
    embeddings_ids = np.array(list(data_dict.keys()))
    pil_image = Image.open(image.file)
    features = image_processor.process_img(pil_image)
    embeddings = model(features)
    embeddings = embeddings.view(embeddings.size(0), -1)
    embeddings = np.array(list(embeddings.tolist()), dtype='float32')
    distance, index_pos = index.search(embeddings.reshape(1, -1), 4)
    similar_images = []
    for similar_image in index_pos:
        similar_images.append(embeddings_ids[similar_image])

    return JSONResponse(content={
    "similar_index": list(similar_images[0]),
        })
    

if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)

