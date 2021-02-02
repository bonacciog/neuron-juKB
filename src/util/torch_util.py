#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Musixmatch AI API
#
# Copyright (c) 2020 Musixmatch spa
#

import os
import torch

def has_gpu():
    
    has_gpu = torch.cuda.is_available()

    if has_gpu is True:
        os.environ["HAS_GPU"] = '1'
    else:
        os.environ["HAS_GPU"] = '0'

    return