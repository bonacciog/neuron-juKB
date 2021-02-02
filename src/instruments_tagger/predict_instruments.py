
# -*- coding: utf-8 -*-
#
# Musixmatch API API
#
# Copyright (c) 2018-2019 Musixmatch spa
#

import os
import numpy as np
import librosa
from collections import Counter
import datetime
import tempfile, shutil, os, time

import tensorflow.compat.v1.keras as keras
from tensorflow.python.keras.backend import set_session
import tensorflow as tf

SR = 22050

labels_group = ['electric bass', 'acoustic guitar', 'synthesizer', 'drum set', 'fx/processed sound', 'voice',
                'violin', 'piano', 'distorted electric guitar', 'clean electric guitar', 'OTHER']
    
def load_multi_instruments_model(MODEL_PATH, session):
    '''
        Load the model
    '''

    # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
    # Otherwise, their weights will be unavailable in the threads after the session there has been set
    set_session(session)
    model = keras.models.load_model(MODEL_PATH)
    model._make_predict_function()

    return model

def melspectrogram(audio, OFF, DURATION):
    '''
        Get Log Mel-Spectrogram
    '''
    N_MEL_BANDS = 128
    
    y, sr = librosa.load(audio, offset=OFF, duration=DURATION, sr=SR, mono=True)
    
    # zero-pad time series to get 5 seconds duration if needed
    if y.shape[0] < SR * DURATION:
        y = np.pad(y, (0, SR * DURATION), 'constant')
        
    y = y[:SR * DURATION]
    
    # compute log-mel-spectrogram
    melspectr = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL_BANDS, fmax=SR/2)
    logmelspectr = librosa.power_to_db(melspectr**2, ref=1.0)
    
    return logmelspectr

def multi_instr_predict(S, model, session):
    '''
        Do predictions and organize labels
    '''
    X = list()
    X.append(S)
    X = np.array(X).reshape(-1, 128, 216, 1)

    with session.as_default():
        with session.graph.as_default():
            labels = np.round(model.predict(X)[0])
        
            prediction = []

            for i in range(0, len(labels)):
                if labels[i] == 1:
                    prediction.append(labels_group[i])

            return prediction

def merge_predictions(data):

    predictions = []
    start = []
    end = []

    for dic in data:
        predictions.append(dic['prediction'])
        start.append(dic['start'])
        end.append(dic['end'])

    joined_pred = []
    joined_pred.append(predictions[0])

    start2 = []
    end2 = []
    start2.append([start[0]])
    end2.append([end[0]])
    
    for i in range(0, len(predictions)-1):
        pred1 = predictions[i]
        pred2 = predictions[i+1]
        lenght = len(joined_pred)
        
        if set(pred1) == set(pred2):
            end2[len(end2) -1 ][0] = end[i+1]
            
            if not (set(pred1) == set(joined_pred[len(joined_pred)-1])):
                joined_pred.append(pred1)
                
        else:
            start2.append([start[i+1]])
            end2.append([end[i+1]])
            joined_pred.append(pred2)

    # convert seconds to minutes
    #start2 = [str(datetime.timedelta(seconds=i[0]))[2:] for i in start2]
    #end2 = [str(datetime.timedelta(seconds=i[0]))[2:] for i in end2]
    
    res = []

    for i in range(0, len(start2)):
        d = {}
        d['start'] = start2[i]
        d['end'] = end2[i]
        d['prediction'] = joined_pred[i]
        res.append(d)

    return res

def create_temporary_copy(path):
    random_name = str(int(time.time()))
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, random_name)
    shutil.copy2(path, temp_path)
    return temp_path

def process(audioPath, model, session, start_time=None, end_time=None, merge=False):
    '''
        Main
    '''
    DURATION = 5 # seconds

    audioPath = create_temporary_copy(audioPath)
        
    # check start-end times
    if start_time is not None and end_time is not None:
        # only predict between start-end
        if start_time > end_time:
            raise Exception('start time must be less than end time')

        if start_time == 0.0:
            intervals = end_time / DURATION
        else:
            intervals = (end_time - start_time) / DURATION # duration / 5sec
            
    else:
        # do prediction for the whole audio file
        y, sr = librosa.load(audioPath)
        audio_duration = librosa.get_duration(y=y, sr=sr)
        start_time = 0.0
        end_time = audio_duration
        intervals = end_time / DURATION
    
    new_start_time = start_time
    count = 0
    
    result = []
    
    # for each interval, make predictions and save results
    while count < intervals:
        
        S = melspectrogram(audioPath, OFF=new_start_time, DURATION=DURATION)
        
        prediction = multi_instr_predict(S, model, session)
        
        d = {}
        d["start"] = new_start_time
        d["end"] = new_start_time + DURATION
        d["prediction"] = prediction
        result.append(d)
        
        new_start_time = new_start_time + DURATION
        count += 1

    if merge is True:
        result = merge_predictions(result)
    
    return result
    

if __name__ == "__main__":
    # load model
    model = load_multi_instruments_model(MODEL_PATH='/Users/stella/Desktop/multi_instr/model/best_model.h5')

    # predict
    audioPath = '/Volumes/Data/dataset/audio/43430950.mp3'
    start_time = 10
    end_time = 26
    predictions = process(audioPath, model, start_time, end_time)
    print(predictions)