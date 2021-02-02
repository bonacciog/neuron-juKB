import torch
import numpy as np
import os
import torch_neuron
from torchvision import models
import logging

## Enable logging so we can see any important warnings
logger = logging.getLogger('Neuron')
logger.setLevel(logging.INFO)

image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)

## Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)

## Tell the model we are using it for evaluation (not training)
model.eval()

## Analyze the model - this will show operator support and operator count
torch.neuron.analyze_model( model, example_inputs=[image] )

## Now compile the model - with logging set to "info" we will see
## what compiles for Neuron, and if there are any fallbacks
## Note: The "-O2" setting is default in recent releases, but may be needed for DLAMI
##       and older installed environments
model_neuron = torch.neuron.trace(model, example_inputs=[image], compiler_args="-O2")

## Export to saved model
BASE_PATH = '/root'
## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/models", exist_ok=True)
model_neuron.save( BASE_PATH + "/neuronsdk/models/resnet50_neuron.pt")