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
from data_utils import (
    Sample,
    create_squad_examples,
    create_inputs_targets,
    normalize_text,
)

for directory in ["data", "checkpoints"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

train_data_file = "https://gitlab.com/bigirqu/quranqa/-/raw/main/datasets/qrcd_v1.1_train.jsonl?inline=false"
dev_data_file = "https://gitlab.com/bigirqu/quranqa/-/raw/main/datasets/qrcd_v1.1_dev.jsonl?inline=false"
train_data = requests.get(train_data_file)
if train_data.status_code in (200,):
    with open("data/train_ar.jsonl", "wb") as train_file:
        train_file.write(train_data.content)
eval_data = requests.get(dev_data_file)
if eval_data.status_code in (200,):
    with open("data/eval_ar.jsonl", "wb") as eval_file:
        eval_file.write(eval_data.content)


with open("data/train_ar.jsonl") as f:
    raw_train_data = [json.loads(l) for l in f]
with open("data/eval_ar.jsonl") as f:
    raw_eval_data = [json.loads(l) for l in f]


model_name = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
slow_tokenizer = BertTokenizer.from_pretrained(model_name)
if not os.path.exists(f"{model_name}_/"):
    os.makedirs(f"{model_name}_/")
slow_tokenizer.save_pretrained(f"{model_name}_/")
tokenizer = BertWordPieceTokenizer(f"{model_name}_/vocab.txt", lowercase=True)
# ============================================= PREPARING DATASET ======================================================
train_squad_examples = create_squad_examples(
    raw_train_data, "Creating training points", tokenizer
)
x_train, y_train = create_inputs_targets(train_squad_examples)
eval_squad_examples = create_squad_examples(
    raw_eval_data, "Creating evaluation points", tokenizer
)
x_eval, y_eval = create_inputs_targets(eval_squad_examples)
train_data = TensorDataset(
    torch.tensor(x_train[0], dtype=torch.int64),
    torch.tensor(x_train[1], dtype=torch.float),
    torch.tensor(x_train[2], dtype=torch.int64),
    torch.tensor(y_train[0], dtype=torch.int64),
    torch.tensor(y_train[1], dtype=torch.int64),
)
print(f"{len(train_data)} training points created.")
train_sampler = RandomSampler(train_data)
train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
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

# ================================================ TRAINING MODEL ======================================================
# TODO: Continue pretraining
# https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForPreTraining
model = BertForQuestionAnswering.from_pretrained(model_name).to(device=gpu)
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "gamma", "beta"]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay_rate": 0.01,
    },
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay_rate": 0.0,
    },
]

optimizer = torch.optim.Adam(
    lr=1e-5, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters
)

for epoch in range(1, epochs + 1):
    # ============================================ TRAINING ============================================================
    print("Training epoch ", str(epoch))
    training_pbar = tqdm(
        total=len(train_data),
        position=0,
        leave=True,
        file=sys.stdout,
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET),
    )
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(train_data_loader):
        batch = tuple(t.to(gpu) for t in batch)
        (
            input_word_ids,
            input_mask,
            input_type_ids,
            start_token_idx,
            end_token_idx,
        ) = batch
        optimizer.zero_grad()
        loss, _, _ = model(
            input_ids=input_word_ids,
            attention_mask=input_mask,
            token_type_ids=input_type_ids,
            start_positions=start_token_idx,
            end_positions=end_token_idx,
            return_dict=False,
        )
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_steps += 1
        training_pbar.update(input_word_ids.size(0))
    training_pbar.close()
    print(f"\nTraining loss={tr_loss / nb_tr_steps:.4f}")
    torch.save(model.state_dict(), "checkpoints/weights_" + str(epoch) + ".pth")
    # ============================================ VALIDATION ==========================================================
    validation_pbar = tqdm(
        total=len(eval_data),
        position=0,
        leave=True,
        file=sys.stdout,
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
    )
    model.eval()
    eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip is False]
    currentIdx = 0
    count = 0
    for batch in validation_data_loader:
        batch = tuple(t.to(gpu) for t in batch)
        (
            input_word_ids,
            input_mask,
            input_type_ids,
            start_token_idx,
            end_token_idx,
        ) = batch
        with torch.no_grad():
            start_logits, end_logits = model(
                input_ids=input_word_ids,
                attention_mask=input_mask,
                token_type_ids=input_type_ids,
                return_dict=False,
            )
            pred_start, pred_end = (
                start_logits.detach().cpu().numpy(),
                end_logits.detach().cpu().numpy(),
            )

        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[currentIdx]
            currentIdx += 1
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.context[pred_char_start:]
            normalized_pred_ans = normalize_text(pred_ans)
            normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
            if normalized_pred_ans in normalized_true_ans:
                count += 1
        validation_pbar.update(input_word_ids.size(0))
    acc = count / len(y_eval[0])
    validation_pbar.close()
    print(f"\nEpoch={epoch}, exact match score={acc:.2f}")

# ============================================ TESTING =================================================================
data = raw_eval_data[:10]
model.eval()
test_samples = create_squad_examples(data, "Creating test points", tokenizer)
x_test, _ = create_inputs_targets(test_samples)
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
    print("Q: " + test_sample.question)
    print("A: " + pred_ans)
