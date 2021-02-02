import json
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import sentencepiece as spm
import torch
import numpy as np
import pandas as pd

from .rap_tools.lyrics import Lyrics, Lyrics_mod


ALPHA = 5


def get_candidate_scores(candidates, input_ids, phonemes_dict):
    
    lookback = 15
    start_time = time.time()
    text_sofar = []
    
    input_ids = input_ids[0][-lookback:]
    
    start_time = time.time()

    for input_id in input_ids:
        token = phonemes_dict[input_id.item()]['BPE_TRANS']
        text_sofar.append(token)

    text_sofar = " ".join(text_sofar)
    all_rl = []
    
    for index , prob in candidates:

        candidate_trans = phonemes_dict[index.item()]['BPE_TRANS']
        candidate_bpe = phonemes_dict[index.item()]['BPE']
        

        cand_plus_lyric = text_sofar + candidate_trans + ' '

        l = Lyrics_mod(precomputed=cand_plus_lyric, language='en-us', lookback=lookback)

        rl = l.get_RL()

        all_rl.append(rl)

    sum_new_prob = 0
    num_probs = []

    for (index , prob), rl in zip(candidates,all_rl):
        original = phonemes_dict[index.item()]['BPE']
        token = phonemes_dict[index.item()]['BPE_TRANS']
        new_prob = prob.item() * (1 + ALPHA * rl)
        num_probs.append(new_prob)
        
        #print(index, original, token, prob, new_prob)

    sum_new_prob=sum(num_probs)
    num_probs = np.array(num_probs)
    num_probs_normalized = num_probs/sum_new_prob

    distribution = np.random.multinomial(1, num_probs_normalized)
    sampled_word = np.argmax(distribution)

    next_token = candidates[sampled_word][0]
    #print("next_token")
    #print(next_token)
    #print("\n\n\n\n\n")
    return next_token




    