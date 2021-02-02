#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Musixmatch AI API
#
# @author Maria Stella Tavella at musixmatch dot com
# @author Pierfrancesco Melucci at musixmatch dot com
# Copyright (c) 2020 Musixmatch spa
#

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD,Adam


def create_conv_model(shape, nh=64):

    model = Sequential()

    model.add( Conv2D( nh, (3,3), input_shape=shape ) )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))

    model.add( Conv2D(nh, (2,2)) )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(2))

    model.add(Activation('sigmoid'))

    print(model.summary())
    
    return model