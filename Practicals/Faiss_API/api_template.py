# %%
import pickle
import uvicorn
from fastapi import FastAPI
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

# %%
'''
Running the image processing script with supplied image
'''
image_processor.process_img()
# %%

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
        Replaces the final layer with a 1000 way neuron for extraction to FAISS
    
    forward():
        Initiates the forward pass
        
    """       
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        #self.resnet50 = model
        self.resnet50.fc = torch.nn.Linear(2048,13)

    def forward(self, X):
        return F.softmax(self.resnet50(X))

##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the image model   #
##############################################################
        
        self.decoder = decoder

    def forward(self, image):
        x = self.main(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

try:
    model = ItemFeatureExtractor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    checkpoint = torch.load('final_weights/weights.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#################################################################
# TO DO                                                          #
# Load the Feature Extraction model. Above, we have initialized #
# a class that inherits from nn.Module, and has the same        #
# structure as the model that you used for training it. Load    #
# the weights in it here.                                       #
#################################################################
    
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:
##################################################################
# TODO                                                           #
# Load the FAISS model. Use this space to load the FAISS model   #
# which is was saved as a pickle with all the image embeddings   #
# fit into it.                                                   #
##################################################################
    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")


app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

  
@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)

    ################################################################
    # TODO                                                         #
    # Process the input and use it as input for the feature        #
    # extraction model image. File is the image that the user      #
    # sent to your API. Apply the corresponding methods to extract #
    # the image features/embeddings.                               #
    ################################################################

    return JSONResponse(content={
    "features": "", # Return the image embeddings here
    
        })
  
@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    
    #####################################################################
    # TODO                                                              #
    # Process the input  and use it as input for the feature            #
    # extraction model.File is the image that the user sent to your API #   
    # Once you have feature embeddings from the model, use that to get  # 
    # similar images by passing the feature embeddings into FAISS       #
    # model. This will give you index of similar images.                #            
    #####################################################################

    return JSONResponse(content={
    "similar_index": "", # Return the index of similar images here
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)
# %%
