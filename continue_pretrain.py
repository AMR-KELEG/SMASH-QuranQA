# Plan
# - Create batches of masked samples
# - Find a way to compute loss and backpropagate it
#     - (Check other posts)
# - Find a way to select the hyperparams (learning rate, number of epochs, weight decay, ....)
#     - Start with the basic configuration
# - Check the effect of the first trial on the performance
# - Use multitask learning in the sense of predicting the surah as well? As a sequence classification task?

import json
import os
import re
import sys
import requests
import string
import numpy as np
from colorama import Fore
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from transformers import BertTokenizer, BertForQuestionAnswering, BertForMaskedLM
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from settings import gpu, epochs, max_seq_length, batch_size

train_data_file = "https://gitlab.com/bigirqu/quranqa/-/raw/main/datasets/qrcd_v1.1_train.jsonl?inline=false"
dev_data_file = "https://gitlab.com/bigirqu/quranqa/-/raw/main/datasets/qrcd_v1.1_dev.jsonl?inline=false"
train_data = requests.get(train_data_file)
if train_data.status_code in (200,):
    with open("train_ar.jsonl", "wb") as train_file:
        train_file.write(train_data.content)
eval_data = requests.get(dev_data_file)
if eval_data.status_code in (200,):
    with open("eval_ar.jsonl", "wb") as eval_file:
        eval_file.write(eval_data.content)


with open("train_ar.jsonl") as f:
    raw_train_data = [json.loads(l) for l in f]
with open("eval_ar.jsonl") as f:
    raw_eval_data = [json.loads(l) for l in f]


model_name = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
slow_tokenizer = BertTokenizer.from_pretrained(model_name)
# if not os.path.exists(f"{model_name}_/"):
#     os.makedirs(f"{model_name}_/")
# slow_tokenizer.save_pretrained(f"{model_name}_/")
# tokenizer = BertWordPieceTokenizer(f"{model_name}_/vocab.txt", lowercase=True)
tokenizer = slow_tokenizer

# TODO: Have a better way to tokenize quran
with open("quran.txt", "r") as f:
    quran_verses = [l for l in f]

model = BertForMaskedLM.from_pretrained(model_name)
verse_index = 100
verse_text = quran_verses[verse_index]


def get_masked_verse(verse_text, tokenizer):
    mask_token = tokenizer.mask_token
    verse_masked_text = tokenizer.tokenize(verse_text)

    token_ids = []
    cur_token_id = 0
    for token in verse_masked_text:
        if not token.startswith("##"):
            cur_token_id += 1
        token_ids.append(cur_token_id)

    from random import randrange

    # Pick a random token (whole token) to mask
    irand = randrange(0, cur_token_id + 1)

    return " ".join(
        [
            t if token_id != irand else mask_token
            for token_id, t in zip(token_ids, verse_masked_text)
        ]
    )


verse_masked_text = get_masked_verse(verse_text, tokenizer)
encoded_tokens = tokenizer(verse_masked_text, return_tensors="pt")
logits = model(**encoded_tokens, return_dict=True).logits.reshape(-1, 30000)
pred = logits.argmax(axis=1)
print(verse_masked_text)
print(verse_text)
