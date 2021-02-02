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
import pandas

# pytorch, neuron
import torch
import torch_neuron
from util.torch_util import has_gpu

from transformers import GPT2Tokenizer
from generate_lyrics.MXMGPT2LMHeadModel import MXMGPT2LMHeadModel
from generate_lyrics.inference import inference

BASE_PATH = '/root'
## Create test dir if not
os.makedirs(BASE_PATH + "/neuronsdk", exist_ok=True)
os.makedirs(BASE_PATH + "/neuronsdk/models", exist_ok=True)

## Enable logging so we can see any important warnings
logger = logging.getLogger('Neuron')
logger.setLevel(logging.INFO)

# check if gpu is available
has_gpu()
logger.info("has gpu:%s" % os.getenv('HAS_GPU', '0'))

model_path = BASE_PATH + "/generation_lyrics/en"
# gpt2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
# gpt2 model
model = MXMGPT2LMHeadModel.from_pretrained(model_path)
# ipa phonemes table
dataframe_ipa_path = os.path.join(model_path, "ipa_dataframe.pkl")
trans_phonemes_table = pandas.read_pickle(dataframe_ipa_path).to_dict('index')
# inference 
prompt_text = "Hello how are you?"
run_inference = False
if run_inference:
    generated_text = inference(prompt_text=prompt_text, 
                                tokenizer=tokenizer, 
                                model=model, 
                                phonemes_dict=trans_phonemes_table, 
                                length=100,
                                temperature=1.0,
                                top_k=8,
                                top_p=0.9,
                                repetition_penalty=1.0,
                                do_sample=True,
                                do_rhyming=True,
                                stop_token='<endoflyrics>',
                                no_repeat_ngram_size=3)
    
    logger.info("in text:%s out text:%s" % (prompt_text,generated_text))

encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

example_inputs = encoded_prompt

## Analyze the model - this will show operator support and operator count
torch.neuron.analyze_model( model, example_inputs = example_inputs )

## Now compile the model - with logging set to "info" we will see
## what compiles for Neuron, and if there are any fallbacks
## Note: The "-O2" setting is default in recent releases, but may be needed for DLAMI
##       and older installed environments
model_neuron = torch.neuron.trace(model, example_inputs=example_inputs, compiler_args="-O2")
model_neuron.save( BASE_PATH + "/neuronsdk/models/gpt2_generate_lyrics_neuron.pt")