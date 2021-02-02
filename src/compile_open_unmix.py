#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Musixmatch AI API
#
# @author Loreto Parisi at musixmatch dot com
# Copyright (c) 2020 Musixmatch spa
#

import os
import logging

# pytorch, neuron
import torch
import torch_neuron
from util.torch_util import has_gpu

from open_unmix_torch import audioseparation


BASE_PATH = '/root'
## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/models", exist_ok=True)

## Enable logging so we can see any important warnings
logger = logging.getLogger('Neuron')
logger.setLevel(logging.INFO)

# check if gpu is available
has_gpu()
logger.info("has gpu:%s" % os.getenv('HAS_GPU', '0'))

## TODO: your Pytorch tokenizer, model here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
targets = ['vocals', 'drums', 'bass', 'other']
model_name = BASE_PATH + '/open_unmix_pytorch_models/UMX-HQ'
unmix_target_list = audioseparation.get_models(targets, model_name, device)

new_targets = ['vocals', 'drums', 'bass', 'other']
new_unmix_target_list = []
for t in new_targets:
    _id = targets.index(t)
    new_unmix_target_list.append(unmix_target_list[_id])
audioseparation.process(BASE_PATH + '/audio/test.wav', new_targets, new_unmix_target_list)

# vocals OpenUnmix model
model = unmix_target_list[0]
# audio preprocessing
input = audioseparation.audio_preprocessing(BASE_PATH + '/audio/test.wav')
# Tensor (88175, 2)
example_inputs = input

## Analyze the model - this will show operator support and operator count
#torch.neuron.analyze_model( model, example_inputs = example_inputs )

## Now compile the model - with logging set to "info" we will see
## what compiles for Neuron, and if there are any fallbacks
## Note: The "-O2" setting is default in recent releases, but may be needed for DLAMI
##       and older installed environments
#model_neuron = torch.neuron.trace(model, example_inputs=example_inputs, compiler_args="-O2")
#model_neuron.save( BASE_PATH + "/neuronsdk/models/bert_explicit_neuron.pt")