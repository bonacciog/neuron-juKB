import pandas as pd
import re


import codecs
import re
import numpy as np
import os
import heapq
import datetime as dt
import json

from lyrics import Lyrics


df = pd.read_pickle("dataset_compressed.pkl")
df = df[df["RD_score"] > 1.15]
df = df[df["RD_score"] < 1.7]
print(df.shape)

total_len = len(df)
last_train_gpt2_index = int(total_len * 14/15)
with open("finetune_train_gpt2.txt", "a") as myfile:
	for i in range(last_train_gpt2_index):
		#print(i)
		text = df.iloc[i]["lyrics_body"]
		author = df.iloc[i]["artist_name"]
		myfile.write("Author: " + author + "\n")
		myfile.write(text + "\n" + "<|endoftext|>" + "\n") 


with open("finetune_valid_gpt2.txt", "a") as myfile:
	for i in range(last_train_gpt2_index,total_len):
		#print(i)
		text = df.iloc[i]["lyrics_body"]
		author = df.iloc[i]["artist_name"]
		myfile.write("Author: " + author + "\n")
		myfile.write(text + "\n" + "<|endoftext|>" + "\n") 






