import pickle
import uvicorn
import fastapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch.nn.functional as F
import torch
import torch.nn as nn
from pydantic import BaseModel
import image_processor
from torchvision import datasets, models
import faiss
import numpy as np
import json

## Load the feature extractor

class ItemFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        #self.resnet50 = model
        self.resnet50.fc = torch.nn.Linear(2048,13)

    def forward(self, X):
        return F.softmax(self.resnet50(X))

try:
    model = ItemFeatureExtractor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    checkpoint = torch.load('final_model/weights.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model = nn.Sequential(*list(model.resnet50.children())[:-1])

#################################################################
# TO DO                                                          #
# Load the Feature Extraction model. Above, we have initialized #
# a class that inherits from nn.Module, and has the same        #
# structure as the model that you used for training it. Load    #
# the weights in it here.                                       #
#################################################################
    
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")


## Load the FAISS index
try:
    index = faiss.IndexFlatL2(2048)
    with open('index.pickle', 'rb') as file:
        index = pickle.load(file)
##################################################################
# TODO                                                           #
# Load the FAISS model. Use this space to load the FAISS model   #
# which is was saved as a pickle with all the image embeddings   #
# fit into it.                                                   #
##################################################################
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

    ################################################################
    # TODO                                                         #
    # Process the input and use it as input for the feature        #
    # extraction model image. File is the image that the user      #
    # sent to your API. Apply the corresponding methods to extract #
    # the image features/embeddings.                               #
    ################################################################

    return JSONResponse(content={
    "features": list(embeddings[0]) # Return the image embeddings here
    
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
    #print(text)
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
    D, I = index.search(embeddings.reshape(1, -1), 4)
    similar_images = []
    for similar_image in I:
        similar_images.append(embeddings_ids[similar_image])
    
    
    #####################################################################
    # TODO                                                              #
    # Process the input  and use it as input for the feature            #
    # extraction model.File is the image that the user sent to your API #   
    # Once you have feature embeddings from the model, use that to get  # 
    # similar images by passing the feature embeddings into FAISS       #
    # model. This will give you index of similar images.                #            
    #####################################################################

    return JSONResponse(content={
    "similar_index": list(similar_images[0]), # Return the index of similar images here
        })
    

if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)

