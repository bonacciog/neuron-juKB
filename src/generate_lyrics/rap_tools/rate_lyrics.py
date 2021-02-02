import pandas
import re


import codecs
import re
import numpy as np
import os
import heapq
import datetime as dt
import json

from lyrics import Lyrics

def clean(text):
	new_text = ""
	bad_words = ['chorus', '[', ']', '{', '}', ' :', '::', 'verse:', 'Verse: ', 'chrous', 'Chorus', 'Chorus:', 'chorus:', 'CHORUS', '^^', 'Repeat x', 'repeat x']
	text = text.replace('_NL_','\n')
	text = re.sub("[\(\[].*?[\)\]]", "", text)
	for line in text:
		if not any(bad_word in line for bad_word in bad_words):
			line = line.replace('+', '')
			line = line.replace('/', '')
			line = line.replace("\\", '')
			line = line.replace("`", '\'')
			new_text = new_text + line
	#print(new_text)
	return new_text





def compute_RD(text_original):
    l = Lyrics(text=text_original, language='en-us', lookback=15)
    rl = l.get_avg_rhyme_length()
    return rl


file = "../dataset/20190806_hiphoprap_dataset_en.csv"
df = pandas.read_csv(file, delimiter = '\t')
df = df[df["lyrics_language"] == "en"]
df = df.drop_duplicates()
df['word_count'] = df['lyrics_body'].apply(lambda x: len(str(x).split(" ")))
df = df[df["word_count"] > 100]
df['lyrics_body'] = df['lyrics_body'].apply(lambda x: clean(x))
df['RD_score'] = df['lyrics_body'].apply(lambda x: compute_RD(x))


#print(df.iloc[3]["lyrics_body"])
#print("--------------------------------------------------------------------------")
#print(clean(df.iloc[3]["lyrics_body"]))
#print(df["lyrics_body"])
print(df["RD_score"])
print(df.RD_score.describe())
df.to_pickle("dataset_compressed.pkl")