#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Musixmatch AI API
#
# @author Loreto Parisi at musixmatch dot com
# Copyright (c) 2020 Musixmatch spa
#

import joblib
import os
import sklearn

from util.inference_util import print_and_inference, print_confusion_matrix

BASE_PATH = '/root'
## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/models", exist_ok=True)

# load embedding and regressor
idf_vectorizer = joblib.load(os.path.join(BASE_PATH,'explicit_lyrics_classifier/tfidf_vectorizer_explicitness.jl'))
classifier = joblib.load(os.path.join(BASE_PATH,'explicit_lyrics_classifier/logistic_regression_explicitness.jl'))

int2label = {0 : "NOT_EXPLICIT", 1 : "EXPLICIT"}

## we expect "explicit", "not_explicit", "explicit"
sequences = ["This is a shitty sentence of fucking horrible language" , "This is not an explicit sentence with very bad language", "Fuck you"]
y_true = ["EXPLICIT","NOT_EXPLICIT","EXPLICIT"]
y_preds = []

print("Inference of " + str(len(sequences)) + " examples...")
for i, sequence in enumerate(sequences): #add tqdm with many examples
    print("\n----sequence n."+str(i)+"----")
    print(sequence)
    ## single input

    test = idf_vectorizer.transform([sequence])
    predictions = classifier.predict_proba(test) #probs
    pred = print_and_inference(predictions[0],int2label)
    y_preds.append(int2label[pred])

print("\n----Confusion matrix----")
print_confusion_matrix(y_true=y_true, y_preds = y_preds, labels=list(int2label.values()), verbose=True )