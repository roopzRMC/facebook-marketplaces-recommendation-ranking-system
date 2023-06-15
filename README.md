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

12 hours of training to cover 150 epochs was required to reach a Training Accuracy of 99%, Validation Accuracy of 52%

Further training could have been carried out which may well have reduced the current level of overfitting observed from the tensorboard performance available here

https://tensorboard.dev/experiment/s1zOX9TjQI2CrE6pYx0IBw/#scalars

Screenshot for reference

![training accuracy]("Practicals/tb_screenshots/training_accuracy.png")

![validation accuracy]("Practicals/tb_screenshots/validation_accuracy.png")




## Creating the training and test sets

```
import split-folders

splitfolders.ratio('pytorch_images', output='pytorch_images_split', seed=42, ratio=(0.7,0.3))
```

Using split folders library I used a 70/30 split across the 13 categories

## Experiments 

> Batch Size

Changing the batch size from 16 to 32 had serious impacts on GPU when the full training set was used - it ran out of memory. On smaller versions of the dataset, training times increased but overfitting led the difference to be too great between training and validation scores

> Training and Test Image Data Set Sizes

Limiting the total number of training images in each class to 50 or less yielded unusable training results with the scores rarely moving above 20%

> Optimiser

SGD Optimiser at a learning rate of 0.01 yielded consistently the best results however the severe level of overfitting could be tempered in future by adding dropout 

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