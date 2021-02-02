#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Musixmatch AI API
#
# @author Loreto Parisi at musixmatch dot com
# Copyright (c) 2020 Musixmatch spa
#

import os, sys
import logging

# LP: force cpu to avoid
# Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HAS_GPU"] = '0'

# tensorflow, neuron
import tensorflow as tf
import tensorflow.neuron as tfn
import tensorflow.compat.v1.keras as keras

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(BASE_PATH, 'spleeter'))
sys.path.insert(0, os.path.join(BASE_PATH, 'spleeter', 'spleeter_audio'))

from util.tensorflow_util import load_tensorflow_shared_session
from spleeter import spleeter_separation_v2


BASE_PATH = '/root'
## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/models", exist_ok=True)

## Enable logging so we can see any important warnings
logger = logging.getLogger('Neuron')
logger.setLevel(logging.INFO)

logger.info("has gpu")

## tensorflow shared session and config
#session, config = load_tensorflow_shared_session()
session = None
config = None

## spleeter model
model = spleeter_separation_v2.Separator("spleeter:2stems", 
    MODEL_DIR= BASE_PATH + "/spleeter_pretrained_models/2stems", 
    session=session, 
    config=config)

## inference test
model.separate_to_file(BASE_PATH + "/audio/test.mp3", 
    BASE_PATH + "/audio", codec="mp3", bitrate='128k', 
    filename_format='{filename}_vocals.{codec}')


# Compile using Neuron
model_dir = BASE_PATH + "/spleeter_saved_models/saved_models_py/2stems/"
compiled_model_dir = os.path.join(BASE_PATH + "/neuronsdk/models", 'spleeter_saved_model')
tfn.saved_model.compile(model_dir, compiled_model_dir)

        
