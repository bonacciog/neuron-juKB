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

# tensorflow, neuron
import tensorflow as tf
import tensorflow.neuron as tfn
import tensorflow.compat.v1.keras as keras

## TODO: your Tensorflow imports
from lyrics_boundary_detection import inference

BASE_PATH = '/root'
## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/models", exist_ok=True)

## Enable logging so we can see any important warnings
logger = logging.getLogger('Neuron')
logger.setLevel(logging.INFO)

model_dir = BASE_PATH + "/lyrics_boundary_detection"

## TODO: your Tensorflow tokenizer, model here
input = "White shirt now red, my bloody nose\nSleepin', you're on your tippy toes\nCreepin' around like no one knows\nThink you're so criminal\n\nBruises on both my knees for you\nDon't say thank you or please\nI do what I want when I'm wanting to\nMy soul, so cynical\n\nSo you're a tough guy\nLike it really rough guy\nJust can't get enough guy\nChest always so puffed guy\nI'm that bad type\nMake your mama sad type\nMake your girlfriend mad tight\nMight seduce your dad type\nI'm the bad guy, duh\n\nI'm the bad guy\n\nI like it when you take control\nEven if you know that you don't\nOwn me, I'll let you play the role\nI'll be your animal\nMy mommy likes to sing along with me\nBut she won't sing this song\nIf she reads all the lyrics\nShe'll pity the men I know\n\nSo you're a tough guy\nLike it really rough guy\nJust can't get enough guy\nChest always so puffed guy\nI'm that bad type\nMake your mama sad type\nMake your girlfriend mad tight\nMight seduce your dad type\nI'm the bad guy, duh\n\nI'm the bad guy\nDuh\n\nI'm only good at bein' bad, bad\n\nI like when you get mad\nI guess I'm pretty glad that you're alone\nYou said she's scared of me?\nI mean, I don't see what she sees\nBut maybe it's 'cause I'm wearing your cologne\n\nI'm a bad guy\n\nI'm-I'm a bad guy\nBad guy, bad guy\nI'm a bad"
input = input.split('\n')
input = [i.replace('\n', '') for i in input]

model = inference.model_load(model_dir)
prediction = inference.process_inference(input, model)
for i, l in enumerate(input):
    if prediction[i] == 1:
        print('1')
    print('0:'+l)

# Compile using Neuron
compiled_model_dir = os.path.join(BASE_PATH + "/neuronsdk/models", 'lyricsboundary_neuron')
tfn.saved_model.compile(model_dir, compiled_model_dir)
