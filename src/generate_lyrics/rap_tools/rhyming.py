import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import sentencepiece as spm
import torch
import numpy as np
import pandas as pd

from rap_tools.lyrics import Lyrics, Lyrics_mod

trans_table = pd.read_pickle("bpe_dataframe.pkl")

def compute_RL(candidate, text_sofar):
    return 34
    
    cadidate_trans = trans_table.loc[trans_table['Class'] == candidate, 'BPE_TRANS'].item()
    cadidate_bpe = trans_table.loc[trans_table['Class'] == candidate, 'BPE'].item()

    if(cadidate_bpe[0] == ' '):
        cand_plus_lyric = lyric_sofar + ' ' + cadidate_trans + ' '
    else:
        cand_plus_lyric = lyric_sofar + cadidate_trans + ' '
    
    l = Lyrics_mod(precomputed=cand_plus_lyric, language='en-us', lookback=15)

    rl = l.get_RL()
    
    return rl

def get_candidate_score(candidates, input_ids):
    rl = compute_RL(candidates, input_ids)
    return rl