#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Musixmatch AI API
#
# @author Loreto Parisi at musixmatch dot com
# Copyright (c) 2020 Musixmatch spa
#

import os
import fasttext
import logging

from util.inference_util import print_and_inference, print_confusion_matrix

BASE_PATH = '/root'
## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/models", exist_ok=True)

model_path = BASE_PATH + '/ft_spam_model_4.bin'

# model
model = fasttext.load_model(model_path)
num_labels = 2
threshold = 0.5

int2label = {0 : "spam", 1 : "verified"}

## we expect "spam", "spam" "verified"
sequences = ["Please go to www.mysite.com" , "xxxxxxxxx", "We're causin' utter devastation When we're stepping to the place"]
y_true = ["spam","spam","verified"]
y_preds = []


print("Inference of " + str(len(sequences)) + " examples...")
for i, sequence in enumerate(sequences): #add tqdm with many examples
    print("\n----sequence n."+str(i)+"----")
    print(sequence)
    
    ## single input

    # be aware to normalize text
    # text must be utf-8 str, lowercase
    normalized_text = sequence.lower()
    predictions = model.predict(normalized_text, k=num_labels)
    pred = print_and_inference(predictions[1],int2label)
    y_preds.append(int2label[pred])

print("\n----Confusion matrix----")
print_confusion_matrix(y_true=y_true, y_preds = y_preds, labels=list(int2label.values()), verbose=True )


# # be aware to normalize text
# # text must be utf-8 str, lowercase
# normalized_text = sequences[0].lower()

# # predict
# prediction = model.predict(normalized_text, k=num_labels)
# # (('__label__SPAM', '__label__VERIFIED'), array([1.00001001e+00, 1.00000034e-05]))
# print(prediction)
# if prediction:
#     lang = str(prediction[0][0])
#     lang = lang.replace('__label__', '')
#     score = prediction[1][0]
#     if score < threshold:
#         lang = None
#     print(lang, score)