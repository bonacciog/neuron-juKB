import pandas as pd
import re


df = pd.read_pickle("dataset_compressed.pkl")
df = df.sort_values(by ='RD_score' , ascending=False)
print(df["RD_score"])
print(df.RD_score.describe())
print(df.iloc[65000]["lyrics_body"])
print(df.iloc[65000]["RD_score"])