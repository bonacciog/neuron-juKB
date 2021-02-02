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

# LP: force cpu to avoid
# Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# tensorflow, neuron
import tensorflow as tf
import tensorflow.neuron as tfn
import tensorflow.compat.v1.keras as keras

## TODO: your Tensorflow imports
from util.tensorflow_util import load_tensorflow_shared_session
from instruments_tagger import predict_instruments

BASE_PATH = '/root'
## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/models", exist_ok=True)

## Enable logging so we can see any important warnings
logger = logging.getLogger('Neuron')
logger.setLevel(logging.INFO)

## TODO: your Tensorflow tokenizer, model here
## tensorflow shared session and config
session, config = load_tensorflow_shared_session()

model_path = BASE_PATH + '/instruments/saved_model'
model = predict_instruments.load_multi_instruments_model(model_path, session)
        

# Compile using Neuron
#model_dir = BASE_PATH + "/"
#compiled_model_dir = os.path.join(BASE_PATH + "/neuronsdk/models", 'spleeter_saved_model')
#tfn.saved_model.compile(model_dir, compiled_model_dir)
