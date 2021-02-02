import os
import time
import torch
import torch_neuron
import json
import numpy as np

from urllib import request

from torchvision import models, transforms, datasets

BASE_PATH = '/root'

## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/images", exist_ok=True)

## Create an image directory containing a small kitten
request.urlretrieve("https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg",
                    BASE_PATH + "/neuronsdk/images/kitten_small.jpg")


## Fetch labels to output the top classifications
request.urlretrieve("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
    BASE_PATH + "/neuronsdk/imagenet_class_index.json")

idx2label = []

with open(BASE_PATH + "/neuronsdk/imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

## Import a sample image and normalize it into a tensor
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

eval_dataset = datasets.ImageFolder(
    os.path.dirname(BASE_PATH + "/neuronsdk/images/"),
    transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normalize,
    ])
)

image, _ = eval_dataset[0]
image = torch.tensor(image.numpy()[np.newaxis, ...])

## Load model
model_neuron = torch.jit.load( BASE_PATH + "/neuronsdk/models/resnet50_neuron.pt" )

## Predict
results = model_neuron( image )

# Get the top 5 results
top5_idx = results[0].sort()[1][-5:]

# Lookup and print the top 5 labels
top5_labels = [idx2label[idx] for idx in top5_idx]

print("Top 5 labels:\n {}".format(top5_labels) )
