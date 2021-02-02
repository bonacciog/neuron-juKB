#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Musixmatch AI API
#
# @author Maria Stella Tavella at musixmatch dot com
# @author Pierfrancesco Melucci at musixmatch dot com
# Copyright (c) 2020 Musixmatch spa
#

from tensorflow import keras
from .similarity import *
from .extract_features import *
import numpy as np

def model_load(model_path):
    # model load
    model = keras.models.load_model(model_path, compile=True )
    return model


def preprocess_lyrics(lyrics_body, kind='lev', min_ssm_size=5, max_ssm_size=128):
    
    # preprocessing data
    data_to_infer = lyrics_body
    
    # do prediction
    if kind=='lev':
        SSM = self_similarity_matrix(data_to_infer, metric=lambda x,y: pow(string_similarity(x,y), 1))
    elif kind=='phon':
        SSM = self_similarity_matrix(data_to_infer, metric=lambda x,y: pow(phonetic_similarity(x,y), 1))
    
    # check for correct size
    ssm_size = SSM.shape[0]
    if ssm_size < min_ssm_size:
        print('Lyric containing a number of lines inferior to minimum admissibile amount: %d' %min_ssm_size)
    elif ssm_size > max_ssm_size:
        print('Lyric containing a number of lines superior to maximum admissibile amount: %d' %max_ssm_size)
    
    ssm_elems = [SSM] * 1
    ssm_tensor = tensor_from_multiple_ssms(multiple_ssms=ssm_elems,pad_to_size=max_ssm_size)
    return ssm_tensor


def process_inference(lyrics_body, model, kind='lev',max_ssm_size=128,min_ssm_size=5):

    ssm_tensor = preprocess_lyrics(lyrics_body)

    # get predictions
    pred = [model.predict( np.reshape(tensor,(1,tensor.shape[0],tensor.shape[1],tensor.shape[2])) ) for tensor in ssm_tensor]
    #  convert probabilities into label 
    predictions = []

    for p in pred:
        p = p[0].tolist()
        #best = np.float(p.max())
        #label = list(p.index(best))
        label = p.index(max(p))
        predictions.append(label)

    return predictions


if __name__ == '__main__':

    lyrics_body = "White shirt now red, my bloody nose\nSleepin', you're on your tippy toes\nCreepin' around like no one knows\nThink you're so criminal\n\nBruises on both my knees for you\nDon't say thank you or please\nI do what I want when I'm wanting to\nMy soul, so cynical\n\nSo you're a tough guy\nLike it really rough guy\nJust can't get enough guy\nChest always so puffed guy\nI'm that bad type\nMake your mama sad type\nMake your girlfriend mad tight\nMight seduce your dad type\nI'm the bad guy, duh\n\nI'm the bad guy\n\nI like it when you take control\nEven if you know that you don't\nOwn me, I'll let you play the role\nI'll be your animal\nMy mommy likes to sing along with me\nBut she won't sing this song\nIf she reads all the lyrics\nShe'll pity the men I know\n\nSo you're a tough guy\nLike it really rough guy\nJust can't get enough guy\nChest always so puffed guy\nI'm that bad type\nMake your mama sad type\nMake your girlfriend mad tight\nMight seduce your dad type\nI'm the bad guy, duh\n\nI'm the bad guy\nDuh\n\nI'm only good at bein' bad, bad\n\nI like when you get mad\nI guess I'm pretty glad that you're alone\nYou said she's scared of me?\nI mean, I don't see what she sees\nBut maybe it's 'cause I'm wearing your cologne\n\nI'm a bad guy\n\nI'm-I'm a bad guy\nBad guy, bad guy\nI'm a bad"
    
    lyrics_body = lyrics_body.split('\n')
    lyrics_body = [i.replace('\n', '') for i in lyrics_body]
    print(lyrics_body)

    model = model_load()

    prediction = process_inference(lyrics_body, model)

    for i, l in enumerate(lyrics_body):
        if prediction[i] == 1:
            print('')
        print(l)