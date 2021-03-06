#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Musixmatch AI API
#
# @author Loreto Parisi at musixmatch dot com
# Copyright (c) 2020 Musixmatch spa
#

import os
import sys
import logging

# pytorch, neuron
import torch
import torch_neuron

from util.torch_util import has_gpu

## TODO: your Pytorch imports
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(BASE_PATH, 'fastspeech'))
sys.path.insert(0, os.path.join(BASE_PATH, 'fastspeech', 'hparams'))
sys.path.insert(0, os.path.join(BASE_PATH, 'fastspeech', 'melgan'))
sys.path.insert(0, os.path.join(BASE_PATH, 'fastspeech', 'text'))
sys.path.insert(0, os.path.join(BASE_PATH, 'fastspeech', 'transformer'))

from fastspeech.inference import fs2_process
from fastspeech.inference import get_FastSpeech2, get_MelGAN

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
fs2modelname = os.path.join(BASE_PATH, 'fastspeech2_models', 'FastMLP.pth.tar')
fastspeech2_model = get_FastSpeech2(fs2modelname)

melgan_modelname = os.path.join(BASE_PATH, 'fastspeech2_models', 'nvidia_tacotron2_LJ11_epoch6400.pt')
melgan_model = get_MelGAN(melgan_modelname)

input = "['hello musixmatch', 'how are you?']"
generated_speech_dir = fs2_process(fastspeech2_model, melgan_model, input)

example_inputs = []

## Analyze the model - this will show operator support and operator count
#torch.neuron.analyze_model( model, example_inputs = example_inputs )

## Now compile the model - with logging set to "info" we will see
## what compiles for Neuron, and if there are any fallbacks
## Note: The "-O2" setting is default in recent releases, but may be needed for DLAMI
##       and older installed environments
#model_neuron = torch.neuron.trace(model, example_inputs=example_inputs, compiler_args="-O2")
#model_neuron.save( BASE_PATH + "/neuronsdk/models/bert_explicit_neuron.pt")