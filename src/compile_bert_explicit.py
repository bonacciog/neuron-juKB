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

from transformers import BertForSequenceClassification, AutoTokenizer


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

# tokenizer
tokenizer = AutoTokenizer.from_pretrained( BASE_PATH + "/explicit-wasabi-official-model/" )

# model
model = BertForSequenceClassification.from_pretrained( BASE_PATH + "/explicit-wasabi-official-model/"  )

## we expect Tensor(1)
sequence = "This is a shitty sentence of fucking horrible language"
## we expect Tensor(0)
sequence = "This is an explicit sentence with very bad language"

# Bit of a hack to get the tokens with the special tokens
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, add_special_tokens=False, return_tensors="pt")

output = model(inputs)[0]
prediction = int(torch.argmax(output))
print(prediction)

example_inputs = inputs

## Analyze the model - this will show operator support and operator count
torch.neuron.analyze_model( model, example_inputs = example_inputs )

## Now compile the model - with logging set to "info" we will see
## what compiles for Neuron, and if there are any fallbacks
## Note: The "-O2" setting is default in recent releases, but may be needed for DLAMI
##       and older installed environments
model_neuron = torch.neuron.trace(model, example_inputs=example_inputs, compiler_args="-O2")
model_neuron.save( BASE_PATH + "/neuronsdk/models/bert_explicit_neuron.pt")