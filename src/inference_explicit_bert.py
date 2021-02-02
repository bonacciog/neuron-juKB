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
from util.inference_util import print_and_inference, print_confusion_matrix

from transformers import BertForSequenceClassification, AutoTokenizer

BASE_PATH = '/root'
## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/models", exist_ok=True)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained( BASE_PATH + "/explicit-wasabi-official-model/" )

# model
model = BertForSequenceClassification.from_pretrained( BASE_PATH + "/explicit-wasabi-official-model/"  )

int2label = {0 : "NOT_EXPLICIT", 1 : "EXPLICIT"}

## we expect "explicit", "not_explicit", "explicit"
sequences = ["This is a shitty sentence of fucking horrible language" , "This is not an explicit sentence with very bad language", "La mia fottuta vita è uno schifo"]
y_true = ["EXPLICIT","NOT_EXPLICIT","EXPLICIT"]
y_preds = []

print("Inference of " + str(len(sequences)) + " examples...")
for i, sequence in enumerate(sequences): #add tqdm with many examples
    print("\n----sequence n."+str(i)+"----")
    print(sequence)
    ## single input
    encoded_sentence = tokenizer.encode(sequence, add_special_tokens=False, return_tensors="pt",max_length=512)
    sentences_classification_logits = model(encoded_sentence)[0]
    predictions = torch.softmax(sentences_classification_logits, dim=1)[0]
    pred = print_and_inference(predictions,int2label)
    y_preds.append(int2label[pred])

print("\n----Confusion matrix----")
print_confusion_matrix(y_true=y_true, y_preds = y_preds, labels=list(int2label.values()), verbose=True )