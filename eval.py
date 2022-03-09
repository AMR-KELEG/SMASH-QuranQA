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
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from settings import gpu, epochs, max_seq_length, batch_size
from data import create_squad_examples, create_inputs_targets

with open("data/eval_ar.jsonl") as f:
    raw_eval_data = [json.loads(l) for l in f]


model_name = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
tokenizer = BertWordPieceTokenizer(f"{model_name}_/vocab.txt", lowercase=True)
# ============================================= PREPARING DATASET ======================================================
eval_squad_examples = create_squad_examples(
    raw_eval_data, "Creating evaluation points", tokenizer
)
x_eval, y_eval = create_inputs_targets(eval_squad_examples)
eval_data = TensorDataset(
    torch.tensor(x_eval[0], dtype=torch.int64),
    torch.tensor(x_eval[1], dtype=torch.float),
    torch.tensor(x_eval[2], dtype=torch.int64),
    torch.tensor(y_eval[0], dtype=torch.int64),
    torch.tensor(y_eval[1], dtype=torch.int64),
)
print(f"{len(eval_data)} evaluation points created.")
eval_sampler = SequentialSampler(eval_data)
validation_data_loader = DataLoader(
    eval_data, sampler=eval_sampler, batch_size=batch_size
)

# ============================================ TESTING =================================================================
model = BertForQuestionAnswering.from_pretrained(model_name).to(device=gpu)
model.load_state_dict(torch.load("checkpoints/weights_8.pth"))
model.eval()

test_samples = create_squad_examples(raw_eval_data, "Creating test points", tokenizer)
# TODO: Fix this!
x_test, y_test = create_inputs_targets(test_samples)
pred_start, pred_end = model(
    torch.tensor(x_test[0], dtype=torch.int64, device=gpu),
    torch.tensor(x_test[1], dtype=torch.float, device=gpu),
    torch.tensor(x_test[2], dtype=torch.int64, device=gpu),
    return_dict=False,
)
pred_start, pred_end = (
    pred_start.detach().cpu().numpy(),
    pred_end.detach().cpu().numpy(),
)

answers = []
ids = []
for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
    test_sample = test_samples[idx]
    offsets = test_sample.context_token_to_char
    start = np.argmax(start)
    end = np.argmax(end)
    pred_ans = None
    if start >= len(offsets):
        continue
    pred_char_start = offsets[start][0]
    if end < len(offsets):
        pred_ans = test_sample.context[pred_char_start : offsets[end][1]]
    else:
        pred_ans = test_sample.context[pred_char_start:]
    answers.append(pred_ans)
    ids.append(test_sample.question_id)
    if pred_ans != test_sample.answer_text:
        print("Q: " + test_sample.question)
        print("A: " + pred_ans)
        print("L: " + test_sample.answer_text)

with open("data/smash_run01.json", "w") as f:
    submission = {
        id: [
            {"answer": answer, "rank": 1, "score": 0.99},
            {"answer": test_sample.answer_text, "rank": 2, "score": 0.01},
        ]
        for id, answer, test_sample in zip(ids, answers, test_samples)
    }
    json.dump(submission, f, ensure_ascii=False)
