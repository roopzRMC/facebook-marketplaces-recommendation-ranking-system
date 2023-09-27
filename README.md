# Facebook Marketplace Recommendation Ranking System

## Practicals - Generating a Pytorch multi-class image classifier

## Gathering the images
The first critical task was to source enough training images across all 13 categories;
* Appliances
* Video Games
* Clothes
* Computers
* DIY Tools
* Health & Beauty
* Home & Garden
* Kids Toys
* Office
* Other
* Phones
* Sports and Leisure
* Books, Film, Music & Games

I used a google chrome extension - FATKUN Batch Image Download which allowed for multiple image types to be downloaded via google image search based on a given search

The process was not fault-free however the average number of images downloaded per category was approximately 230

## System Requirements for Model Training

Moving to a 13 class problem with the pre trained resnet50 model meant that training times using a CPU were incredibly slow.

I used a google cloud based 8 core cpu with Nvidia T4 GPU Vertex AI notebook to deal with model training.

However, availbility of the GPU was not assured and therefore I was forced to spin up an AWS based VM using a G3.Xlarge configuration

## Virtual Machine Configuration

* G3.4Xlarge Virtual Machine with deep learning AMI
* Ubuntu 20.04
* 250 GB Storage
* Installation of jupyter lab, ipywidgets and associated pytorch libraries
* NVIDIA srivers pre-installed

12 hours of training to cover 150 epochs was required to reach a Training Accuracy of 99%, Validation Accuracy of 52%

Further training could have been carried out which may well have reduced the current level of overfitting observed from the tensorboard performance available here

https://tensorboard.dev/experiment/s1zOX9TjQI2CrE6pYx0IBw/#scalars

Screenshot for reference

![training accuracy](https://github.com/roopzRMC/facebook-marketplaces-recommendation-ranking-system/blob/main/tb_screenshots/training_accuracy.png)

![validation accuracy](https://github.com/roopzRMC/facebook-marketplaces-recommendation-ranking-system/blob/main/tb_screenshots/validation_accuracy.png)

![training loss](https://github.com/roopzRMC/facebook-marketplaces-recommendation-ranking-system/blob/main/tb_screenshots/training_loss.png)

![validation loss](https://github.com/roopzRMC/facebook-marketplaces-recommendation-ranking-system/blob/main/tb_screenshots/validation_loss.png)




## Creating the training and test sets

```
import split-folders

splitfolders.ratio('pytorch_images', output='pytorch_images_split', seed=42, ratio=(0.7,0.3))
```

Using split folders library I used a 70/30 split across the 13 categories

## Experiments 

> Batch Size

Changing the batch size from 16 to 32 had serious impacts on GPU when the full training set was used - it ran out of memory. On smaller versions of the dataset, training times increased but overfitting led the difference to be too great between training and validation scores

> Image Size

Lowering the image size required changing the configuration of the first layer to allow for a 64 x 64 image to be ingested

```
self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
self.resnet50.avgpool = nn.AdaptiveAvgPool2d(1)
```

> Additional Image Transforms

Both random horizontal flip and rotations were attempted however when implemented in the train loader had the effect of never converging 

This had the effect of drastically decreasing training time but resulted in a poorly generalising model with validation accuracy not rising above 0.15

> Training and Test Image Data Set Sizes

Limiting the total number of training images in each class to 50 or less yielded unusable training results with the scores rarely moving above 20%

> Optimisers

SGD Optimiser at a learning rate of 0.01 yielded consistently the best results however the severe level of overfitting could be tempered in future by adding dropout 

Lower learning rates were also tested which had the effect of slower training times but not meaningful positive impact on the validation accuracy or loss

A weight decay of 1e4, momentum and nesterov were added which had the effect of quicker training performance but little positive effect on generalisation

Adam, AdamW and RMSProp were also used with an array of parameters and learning rates specified, however the training accuracy never reached that of SGD within 200 epochs. Adam rarely reached above 0.21 for training and 0.07 for validation regardless of learning rate and other parameters such as weight decay

## Feature Extractions

Loading the model from the final epoch 

```
model = ItemFeatureExtractor()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
checkpoint = torch.load('model_test/weights.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

```

Removing the models final layer left a 2048 neuron feature extractor

```
## remove last layer to 2048 feature output
# Remove the last layer
model = nn.Sequential(*list(model.resnet50.children())[:-1])

```

Then, by setting the model to eval mode - it was able to parse an image having processed it via the following transformation function as well adding and additional batch size dimension

```
def process_img(image):
    pil_to_tensor = transforms.ToTensor()
    resize = transforms.Resize((225,225))
    img = Image.open(image).convert('RGB')
    
    features = pil_to_tensor(img)
    features = resize(features)
    ## Additional batch size dimension for processing
    features = torch.unsqueeze(features, dim=0)
    print(features.shape)
    return features
```
## Model Training Script

Please reference ```resnet50_pretrained_13_class_unfreeze_v3_final.ipynb``` as the main model training script 

This must use a GPU as it is optimised for GPU only - the pretrained model is loaded from the ```'NVIDIA/DeepLearningExamples:torchhub'``` and therefore expects a GPU device to be available

## Image Processor

Using the weights file generated from the model training script, the model is instantiated with the parameters and then the checkpoint is loaded of the final weights

```process_img()``` extracts the train loader image processing steps as used in the train loader

The model has the final layer removed to reveal a 2048 neuron output

A loop is called to enter every folder in the training images directory
* For each image in the training directory, an image is processed
* The embeddings are extracted
* The file name is split at the file type and is instantiated as the index
* The index is written as a dictionary'es key and the embedding as the value

## FAISS

Refencing FAISS.ipynb

Note this also relies on a GPU as the classifier has been optimised for GPU use by virtue of the pretrained model being downloaded from the NVIDIA torchhub as specified.

```!pip install faiss-gpu``` is required so that a GPU version of faiss is installed



To create the FAISS search index, a faiss.IndexFlatL2() class is instantiated with a 2048 dimension (as per the shape of the image vector)

The embeddings json file is imported as a dictionary, and converted in to a numpy array of type float32

The embeddings are then added to the index using ```add()``` and the index is the deemed as being trained

Once trained as a test, an image is passed through the image feature extractor to derive a 2048 neuron embedding

The output is gflattened and converted into a float32 numpy array

Using ```index.search(query_vector.reshape(1, -1), 4)```, the query vector is passed through the index with 4 nearest vectors specified as an argument

This return 4 index values as similar embeddings

## API build and FAISS integration
The ```api.py``` script is split in to 4 sections:

1. Loading the model through the item feature extractor class
2. Loading the FAISS index from the ```.pkl``` object
3. Configuring the api to ingest an image and output the 2048-way feature embedding
4. Configuring the api to ingest and image and suggest similar images from the FAISS index

### Item feature extractor

The class is initiated with a load of the pretrained resnet50 model from torchhub with the optimised NVIDIA version for GPU use

The model is initiatlised and the final weights from the model trainins process are loaded including the optimiser. The last layer is the removed to expose the pennultimate 2048 output layer

### Loading the FAISS index

The pickle file is loaded having specified the size of the index as 2048 to match the neuron output of the feature extractor

### Ingesting the image via the api and outputing the embeddings

A fastapi post operation is leveraged to ingest a request containing an image payload

One the image has been accepted, it is then processed through the ```process_img()``` found by importing image_processor to convert the image to features for the featureextractor model. The features are passed through the model as embeddings. The embeddings are convereted into a JSON serialisable object through coercing the output of the numpy array to ```float64```

The response is viewable through executing ```response.text```

### Ingesting the image via api operation to produce similar image output from FAISS index

Another dastapi post operation is leverage to ingest a request containing an image payload as the embeddings output method

The original embeddings json file is loaded to allow retrieval of the image file names

In addition the embeddings output is then supplied to the FAISS ```index.search``` method to produce 4 similar images. The ids supplied from the output of the index search is the parsed to the embeddings_json object to retrieve the image filenames that are deemed to be simlar

This is an example output


![api response output](https://github.com/roopzRMC/facebook-marketplaces-recommendation-ranking-system/blob/main/Practicals/faiss_screenshots/faiss_api.png)

## Docker Image build

To maintain comaptibility with the cuda optimised version of pytorch and torchvision the ```FROM python:3.9``` docker image is selected

So that the app is appropriately self-contained, a working directory ```WORKDIR /app``` is intantiated

Each required file for the api's operation is copied to the working directory;

* requirements.txt
* api.py
* image_processor.py
* index.pickle
* final_model/weights.pt
* final_embeddings.json

In addition to the UBUNTU requirements of

* ffmeg
* libsm6
* libxext6

a specific pip install command is specified to install the CUDA versions of pytorch libraries


```
RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

```

The remaining additional pip dependencies are specified in the requirements.txt file


## Running the docker image

To ensure that the docker image utilises the gpus required for the api to run the following command is required when running the docker image

```
docker run --gpus all -p 8080:8080 faiss_api
```

## Uploading the docker image to docker hub

Log in to docker hub

```
docker login -u $USERNAME
```

The image must be tagged appropriately prior to being pushed to the docker hub

```
docker tag faiss_api rupertcog/faiss_api
```

The image is then pushed to dockerhub

```
docker push rupertcog/faiss_api:latest
```

## Recommended further experimentation

With additional time I would recommend the following steps

* Data Augmentation
Increasing the size of the dataset may address the generalisation problem the model suffers from thereby redcuing the performance gap between training and validation

* Randomised Search
With additional GPU time, performance could be assessed across different combinations of optimiser, learning rate, weight decay, momentum, batch_size to assess which combination would perform most optimally

* Image size
Experimenting across different images size inputs could additionally reduce training time whilst performing different combination trials of the model

