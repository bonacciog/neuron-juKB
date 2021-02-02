import os, subprocess

import pandas as pd

LANGUAGE = "en"

trans_table = pd.read_pickle("ipa_dataframa_" + LANGUAGE + ".pkl")
print(trans_table)