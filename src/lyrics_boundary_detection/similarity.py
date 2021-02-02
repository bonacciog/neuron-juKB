#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Musixmatch AI API
#
# @author Maria Stella Tavella at musixmatch dot com
# @author Pierfrancesco Melucci at musixmatch dot com
# Copyright (c) 2020 Musixmatch spa
#

import numpy as np
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from numpy.linalg import norm
from functools import reduce
import re
import phonetics

def self_similarity_matrix(items, metric):
    return np.array([[metric(x, y) for x in items] for y in items])

def string_similarity(some, other):
    return Levenshtein().get_sim_score(some, other)

def phonetic_similarity(some, other, use_equivalences=False):
    if some == other:
        return 1.0
    if not some or not other:
        return 0.0

    some_phonetics = phonetics.dmetaphone(some)
    other_phonetics = phonetics.dmetaphone(other)
    if some_phonetics == other_phonetics:
        return 1.0

    pair_wise_similarities = []
    for some_phonetic in some_phonetics:
        if not some_phonetic:
            continue
        for other_phonetic in other_phonetics:
            if not other_phonetic:
                continue
            some_equiv = metaphone_representative(some_phonetic) if use_equivalences else some_phonetic
            other_equiv = metaphone_representative(other_phonetic) if use_equivalences else other_phonetic
            pair_wise_similarities.append(string_similarity(some_equiv, other_equiv))
    return 0.0 if not pair_wise_similarities else max(pair_wise_similarities)