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

from laser import embed
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

# load model parameters
laser_encoder = embed.EncodeLoad(
            buffer_size=100, 
            max_sentences=None, 
            encoder= BASE_PATH + '/laser_models/bilstm.93langs.2018-12-26.pt',
            max_tokens=12000, cpu=True)
## Tell the model we are using it for evaluation (not training)
laser_encoder.model().eval()

## load BPE
bpe = embed.BPELoad(BASE_PATH + '/laser_models/93langs.fcodes', BASE_PATH + '/laser_models/93langs.fvocab')
text = "hello encoder"
tokens = text.split()
codes = bpe.apply( tokens )
logger.info("%s\n%s", tokens, codes)

encoded = embed.EncodeText(laser_encoder, codes[0], buffer_size=10000)
logger.info(encoded)

length = 1024
src_lengths = torch.tensor(torch.from_numpy(encoded[0]), dtype=torch.long)
src_tokens = torch.ones(len(src_lengths), length, dtype=torch.long)
example_inputs = [src_tokens, src_lengths]

## Analyze the model - this will show operator support and operator count
torch.neuron.analyze_model( laser_encoder.model(), example_inputs = example_inputs )
        
