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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# tensorflow, neuron
import tensorflow as tf
import tensorflow.neuron as tfn
import tensorflow.compat.v1.keras as keras

from bert.embedding import get_bert_embedding_model, get_embeddings

BASE_PATH = '/root'
## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/models", exist_ok=True)

## Enable logging so we can see any important warnings
logger = logging.getLogger('Neuron')
logger.setLevel(logging.INFO)

# get tokenizer
bert_model, tokenizer = get_bert_embedding_model()

# load lstm keras model
model_path = BASE_PATH + '/bert_lstm_spam'
model = keras.models.load_model(model_path)

text = "Silent night, holy night\nall is calm, all is bright\n'round yon virgin Mother and Child\nholy infant so tender and mild\nsleep in heavenly peace\nsleep in heavenly peace\nSsilent night, holy night\nshepherds quake at the sight\nglories stream from heaven afar\nheav'nly hosts sing Hallelujah (Hallelujah)\nChrist the Savior is born\nChrist the Savior is born\n\nSilent night, holy night\nSon of God, love's pure light\nradiant beams from Thy holy face\nwith the dawn of redeeming grace\nJesus, Lord, at Thy birth\nJesus, Lord, at Thy birth\n\nChrist the Savior is born\nChrist the Savior is born\nChris is born\nHallelujah"

encoded = get_embeddings(tokenizer, bert_model, text)
pred = model.predict(encoded)

print(pred)


# Compile using Neuron
model_dir = BASE_PATH + "/spam-bert-lstm-model"
compiled_model_dir = os.path.join(BASE_PATH + "/neuronsdk/models", 'spam_bert_lstm_saved_model')
#tfn.saved_model.compile(model_dir, compiled_model_dir)
