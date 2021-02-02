#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Musixmatch AI API
#
# @author Loreto Parisi at musixmatch dot com
# Copyright (c) 2020 Musixmatch spa
#

import torch
import torch.nn as nn
import numpy as np
import os, sys
import re, tempfile
from string import punctuation
from shutil import rmtree

import hparams as hp
from g2p_en import G2p
from wavfile import write
from melgan.model import generator
from fastspeech2 import FastSpeech2
from text import text_to_sequence

def get_FastSpeech2(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = os.path.join(model_path)
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
    model.requires_grad = False
    model.eval()
    model = model.to(device)
    return model

def get_MelGAN(melgan_modelname):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    melgan = generator.Generator(hp.n_mel_channels)
    checkpoint = torch.load(melgan_modelname, map_location="cpu")
    melgan.load_state_dict(checkpoint["model_g"])
    melgan.eval(inference=True)
    melgan = melgan.to(device)

    return melgan

def synthesize(model, melgan, text, prefix=''):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)

    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len)

    mel_torch = mel.transpose(1, 2)
    mel_postnet_torch = mel_postnet.transpose(1, 2)
    mel = mel[0].cpu().transpose(0, 1)
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1)
    f0_output = f0_output[0].cpu().numpy()
    energy_output = energy_output[0].cpu().numpy()

    if not os.path.exists(hp.test_path):
        os.makedirs(hp.test_path)

    if melgan is not None:
        with torch.no_grad():
            wav = melgan.inference(mel_torch).cpu().numpy()
            wav = wav.astype('int16')
            #ipd.display(ipd.Audio(wav, rate=hp.sampling_rate))
            # save audio file
            write(os.path.join(GENERATED_SPEECH_DIR, prefix + '.wav'), hp.sampling_rate, wav)

def preprocess(text):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    text = text.rstrip(punctuation)

    g2p = G2p()
    phone = g2p(text)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{'+ '}{'.join(phone) + '}'
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    phone = phone.replace('}{', ' ')

    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])

    return torch.from_numpy(sequence).long().to(device)

def playlist_dir_create(TMP_DIR):

    GENERATED_SPEECH_DIR = os.path.join(TMP_DIR, next(tempfile._get_candidate_names()))

    if not os.path.exists(GENERATED_SPEECH_DIR):
        os.mkdir(GENERATED_SPEECH_DIR)
    else:
        rmtree(GENERATED_SPEECH_DIR)
        os.mkdir(GENERATED_SPEECH_DIR)

    return GENERATED_SPEECH_DIR

def fs2_process(model, melgan, sentences):

    TMP_DIR = tempfile.gettempdir()
    global GENERATED_SPEECH_DIR
    GENERATED_SPEECH_DIR = playlist_dir_create(TMP_DIR)

    for i, sentence in enumerate(sentences):
        phone = preprocess(sentence)
        
        with torch.no_grad():
            synthesize(model, melgan, phone, prefix=str(i))

    return GENERATED_SPEECH_DIR
