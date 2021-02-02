import os, subprocess

import pandas as pd
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
LANGUAGE = "en"

ids = list([262,326, 340,428, 465, 607, 616, 661, 683])
tokens = tokenizer.convert_ids_to_tokens(ids)

print(tokens)