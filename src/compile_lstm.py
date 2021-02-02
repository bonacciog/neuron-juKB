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

from lstm.neuronlstm import NeuronLSTM
from util.torch_util import has_gpu

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

length = 160

src_lengths = torch.tensor(
    [160, 140, 120, 100, 80, 60, 40, 20], dtype=torch.long)
src_tokens = torch.ones(len(src_lengths), length, dtype=torch.long)

model = NeuronLSTM()

example_inputs = [src_tokens, src_lengths]
results = torch.neuron.analyze_model(model, example_inputs=example_inputs)

print(results)