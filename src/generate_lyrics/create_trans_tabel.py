import os, subprocess

import pandas as pd
from transformers import GPT2Tokenizer

#cmd = u'espeak -xq -v%s -f %s > %s' % (language, fname, fname2)

tokenizer = GPT2Tokenizer.from_pretrained("/root/generation_lyrics/en/")
LANGUAGE = "en"

df = pd.DataFrame(columns=['Class', 'BPE', 'BPE_TRANS'])
ids = list(range(len(tokenizer)))
tokens = tokenizer.convert_ids_to_tokens(ids)

for id, token in zip(ids, tokens):
    
    filtered_token = token.replace("Ġ", "")
    command = "espeak -xq -v " + LANGUAGE + " '" + filtered_token + "'"
    command = command.strip()
    try:
        bpe_trans = os.popen(command).readlines()[0].strip()
    except:
        bpe_trans = ""
        
    
    df.loc[id] = [id] + [token] + [bpe_trans]

print(df.head())
print(df)

df.to_csv("ipa_dataframa_" + LANGUAGE + ".csv")
df.to_pickle("ipa_dataframa_" + LANGUAGE + ".pkl")
