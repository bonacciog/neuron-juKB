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

model_path = BASE_PATH + "/hatespeech-official-model"

## model config
config = AutoConfig.from_pretrained(
        model_path,
        cache_dir=None,
        num_labels=10
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
sequence_0 = "You sound like a bitch, bitch fuckin\n\nShut the fuck up!\nWhen your fans become your haters\nYou done?\nFuckin\' beard\'s weird"
## we expect "none"
sequence_1 = "I'm a good boy"

## single input
encoded_sentence = tokenizer.encode(sequence_0, add_special_tokens=False, return_tensors="pt")
sentences_classification_logits = model(encoded_sentence)[0]
predictions = torch.softmax(sentences_classification_logits, dim=1).tolist()[0]
## other behavior religion disability class physical race ethnicity gender sexual_orientation 
## [4.161889955867082e-05, 0.9997691512107849, 1.7333959476673044e-05, 8.499081559421029e-06, 2.4943363314378075e-05, 1.9793131286860444e-05, 5.690664329449646e-05, 1.1574138625292107e-05, 1.3512323675968219e-05, 3.684896000777371e-05]
## int2label = {0 : "other" ,1 : "behavior",2: "religion" ,3: "disability", 4 : "class", 5 : "physical",6 : "race", 7 : "ethnicity", 8 : "gender" , 9 : "sexual_orientation" }
print(predictions)

example_inputs = encoded_sentence
## Analyze the model - this will show operator support and operator count
torch.neuron.analyze_model( model, example_inputs = example_inputs )

## Now compile the model - with logging set to "info" we will see
## what compiles for Neuron, and if there are any fallbacks
## Note: The "-O2" setting is default in recent releases, but may be needed for DLAMI
##       and older installed environments
model_neuron = torch.neuron.trace(model, example_inputs=example_inputs, compiler_args="-O2")
model_neuron.save( BASE_PATH + "/neuronsdk/models/hatespeech_bert_neuron.pt")