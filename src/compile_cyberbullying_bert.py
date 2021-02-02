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

from transformers import (
    AutoConfig,
    BertTokenizerFast,
    PreTrainedTokenizer,
    BertForSequenceClassification
)

# pytorch, neuron
import torch
import torch_neuron
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

model_path = BASE_PATH + "/sexism-racism-official-model"

## model config
config = AutoConfig.from_pretrained(
        model_path,
        cache_dir=None,
        num_labels=3
    )

## tokenizer
tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-multilingual-cased",
        cache_dir=None,
        use_fast=True
    )

## model
model = BertForSequenceClassification.from_pretrained(
        model_path,
        from_tf=False,
        config=config,
        cache_dir=None,
    )

## we expect "sexism"
sequence_0 = "I'm sexual molester"
## we expect "none"
sequence_1 = "I'm a good boy"

## single input
encoded_sentence = tokenizer.encode(sequence_0, add_special_tokens=False, return_tensors="pt")
sentences_classification_logits = model(encoded_sentence)[0]
predictions = torch.softmax(sentences_classification_logits, dim=1).tolist()[0]
## sexism racism none [0.5053430795669556, 0.323892742395401, 0.17076420783996582]
## int2label = {0 : "none", 1 : "sexism", 2 : "racism"}
print(predictions)

example_inputs = encoded_sentence
## Analyze the model - this will show operator support and operator count
torch.neuron.analyze_model( model, example_inputs = example_inputs )

## Now compile the model - with logging set to "info" we will see
## what compiles for Neuron, and if there are any fallbacks
## Note: The "-O2" setting is default in recent releases, but may be needed for DLAMI
##       and older installed environments
model_neuron = torch.neuron.trace(model, example_inputs=example_inputs, compiler_args="-O2")
model_neuron.save( BASE_PATH + "/neuronsdk/models/cyberbullying_bert_neuron.pt")