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
from util.inference_util import print_and_inference, print_confusion_matrix

BASE_PATH = '/root'
## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/models", exist_ok=True)

model_path = BASE_PATH + "/hate-speech-offensive-official-model"

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

int2label = {0 : "hatespeech" , 1 : "offensive", 2 : "neither"}

## we expect "offensive", "hatespeech", "hatespeech"
sequences = ["That ugly boy is orrible" , "You are a nigga", "Gay people are trash"]
y_true = ["offensive","hatespeech","hatespeech"]
y_preds = []

print("Inference of " + str(len(sequences)) + " examples...")
for i, sequence in enumerate(sequences): #add tqdm with many examples
    print("\n----sequence n."+str(i)+"----")
    print(sequence)
    ## single input
    encoded_sentence = tokenizer.encode(sequence, add_special_tokens=False, return_tensors="pt",max_length=512)
    sentences_classification_logits = model(encoded_sentence)[0]
    predictions = torch.softmax(sentences_classification_logits, dim=1)[0]
    ## hatespeech offensive neither [0.16904698312282562, 0.8296677470207214, 0.0012852761428803205]
    pred = print_and_inference(predictions,int2label)
    y_preds.append(int2label[pred])

print("\n----Confusion matrix----")
print_confusion_matrix(y_true=y_true, y_preds = y_preds, labels=list(int2label.values()), verbose=True )