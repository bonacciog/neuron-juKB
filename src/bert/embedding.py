#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Musixmatch AI API
#
# @author Loreto Parisi at musixmatch dot com
# Copyright (c) 2020 Musixmatch spa
#
import numpy as np
from transformers import BertTokenizerFast
from transformers.modeling_bert import BertModel as TFBertModel


def get_bert_embedding_model():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    model = TFBertModel.from_pretrained("bert-base-multilingual-cased")
    return model, tokenizer


def get_embeddings(tokenizer, model, text_array):

    if not isinstance(text_array, list):
        text_array = [text_array]

    embeddings = []

    for text in text_array:
        text = text.replace('\n', '__CLRF__')
        encoded_input = tokenizer(text, return_tensors='tf', add_special_tokens=True,
                        max_length=512, truncation=True, pad_to_max_length=True, return_attention_mask=False) #True
        outputs = model(encoded_input)
        last_hidden_states = outputs[0][0]
        embeddings.append(np.array(last_hidden_states))
    
    embeddings = np.array(embeddings)

    return embeddings


def get_embeddings_single(tokenizer, model, text):

    text = text.replace('\n', '__CLRF__')

    encoded_input = tokenizer.encode(text, return_tensors='tf', add_special_tokens=True, 
                        max_length=512, truncation=True, pad_to_max_length=True, return_attention_mask=False)

    tokens = encoded_input

    outputs = model(encoded_input)
    last_hidden_states = outputs[0][0]
    embedding = np.array([last_hidden_states])

    return tokens, embedding


if __name__ == '__main__':

    # get tokenizer
    bert_model, tokenizer = get_bert_embedding_model()

    text_array = ['hi']
    embeddings = get_embeddings(tokenizer, bert_model, text_array)
    print(embeddings.shape)