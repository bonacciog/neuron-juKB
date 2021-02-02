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

# pytorch
import torch
from util.inference_util import print_and_inference, print_confusion_matrix

BASE_PATH = '/root'
## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/models", exist_ok=True)

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

int2label = {0 : "other" ,1 : "behavior",2: "religion" ,3: "disability", 4 : "class", 5 : "physical",6 : "race", 7 : "ethnicity", 8 : "gender" , 9 : "sexual_orientation" }

## we expect "physical", "race", "sexual_orientation"
sequences = ["That ugly boy is orrible" , "You are a nigga", "Gay people are trash"]
y_true = ["physical","race","sexual_orientation"]
y_preds = []

print("Inference of " + str(len(sequences)) + " examples...")
for i, sequence in enumerate(sequences): #add tqdm with many examples
    print("\n----sequence n."+str(i)+"----")
    print(sequence)
    ## single input
    encoded_sentence = tokenizer.encode(sequence, add_special_tokens=False, return_tensors="pt",max_length=512)
    sentences_classification_logits = model(encoded_sentence)[0]
    predictions = torch.softmax(sentences_classification_logits, dim=1)[0]
    ## other behavior religion disability class physical race ethnicity gender sexual_orientation 
    ## [4.161889955867082e-05, 0.9997691512107849, 1.7333959476673044e-05, 8.499081559421029e-06, 2.4943363314378075e-05, 1.9793131286860444e-05, 5.690664329449646e-05, 1.1574138625292107e-05, 1.3512323675968219e-05, 3.684896000777371e-05]
    pred = print_and_inference(predictions,int2label)
    y_preds.append(int2label[pred])

print("\n----Confusion matrix----")
print_confusion_matrix(y_true=y_true, y_preds = y_preds, labels=list(int2label.values()), verbose=True )