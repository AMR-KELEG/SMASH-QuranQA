import json
import os
import re
import sys

# TODO: Fix this!
sys.path.append("../")
sys.path.append("../quranqa/code/")

import requests
import string
import numpy as np
from colorama import Fore
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from settings import GPU_ID, MODEL_NAME, EPOCHS
from data_utils import create_squad_examples, create_inputs_targets
from quranqa.code.quranqa22_eval import (
    normalize_text,
    remove_prefixes,
    pRR_max_over_ground_truths,
)
import argparse
from utils import find_interrogatives, get_spans
from itertools import groupby
import logging
from model import MultiTaskQAModel

logger = logging.getLogger("Eval")

parser = argparse.ArgumentParser(description="Evaluate the models.")
parser.add_argument("--train", action="store_true")
parser.add_argument(
    "--seed",
    default=0,
    help="The value of the random seed to use.",
)
parser.add_argument(
    "--epoch",
    default=EPOCHS,
    help="The value of the epoch at which the checkpoint was generated.",
)
parser.add_argument("--desc", required=True, help="The description of the model.")

args = parser.parse_args()

if args.train:
    datafile = "data/train_ar.jsonl"
else:
    datafile = "data/eval_ar.jsonl"

with open(datafile, "r") as f:
    raw_eval_data = [json.loads(l) for l in f]


tokenizer = BertWordPieceTokenizer(f"{MODEL_NAME}_/vocab.txt", lowercase=True)
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
validation_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

# ============================================ TESTING =================================================================
model = MultiTaskQAModel(MODEL_NAME).to(device=GPU_ID)
checkpoint_name = (
    f"checkpoints/weights_{args.desc}_seed_{args.seed}_" + str(args.epoch) + ".pth"
)
model.load_state_dict(torch.load(checkpoint_name))
model.eval()

answers = []
ids = []
prrs = []
wrong_answers = []

for i in range(0, len(raw_eval_data), 1):
    batch_data = raw_eval_data[i : i + 1]
    test_samples = create_squad_examples(
        batch_data, f"Creating test points for batch #{i}", tokenizer
    )
    # TODO: Fix this!
    x_test, y_test = create_inputs_targets(test_samples)
    outputs = model(
        torch.tensor(x_test[0], dtype=torch.int64, device=GPU_ID),
        torch.tensor(x_test[1], dtype=torch.float, device=GPU_ID),
        torch.tensor(x_test[2], dtype=torch.int64, device=GPU_ID),
    )

    start_logits, end_logits = outputs[1].split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()

    pred_start, pred_end = (
        start_logits.detach().cpu().numpy(),
        end_logits.detach().cpu().numpy(),
    )

    for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
        # TODO: What is this doing actually?
        test_sample = test_samples[idx]
        try:
            offsets = test_sample.context_token_to_char
        except:
            # TODO: This is a hack added by Amr!
            # Investigate the reason for this!
            offsets = []

        # N.B: This might be an invalid range (i.e: start > end)
        greedy_range = [(1.0, np.argmax(start), np.argmax(end))]

        # Greedily find the most probable 5 candidate spans of answers
        answers_ranges = get_spans(start, end, n_ranges=3) + [(1.0, len(offsets), -1)] + greedy_range

        # Avoid greedy decoding for ranges
        # start = np.argmax(start)
        # end = np.argmax(end)

        pred_answers = []
        pred_char_starts = []

        # Reverse the range of spans considering the most probable one first!
        for prob, start, end in answers_ranges[::-1]:
            pred_ans = None
            # Is the start of the range valid?
            if start < len(offsets):
                pred_char_start = offsets[start][0]
                if end < len(offsets):
                    pred_ans = test_sample.context[pred_char_start : offsets[end][1]]
                else:
                    pred_ans = test_sample.context[pred_char_start:]
            else:
                pred_ans = test_sample.context
                pred_char_start = 0
            pred_answers.append(pred_ans)
            pred_char_starts.append([pred_char_start])

        assert(len(pred_answers)==len(pred_char_starts))
        # TODO: How are char_start used?
        prr = pRR_max_over_ground_truths(
            pred_answers, pred_char_starts, [{"text": a["text"]} for a in test_sample.all_answers]
        )
        answers.append(pred_answers)
        ids.append(test_sample.question_id)
        prrs.append(prr)

        cleaned_pred = normalize_text(remove_prefixes(pred_answers[-1]))
        cleaned_ans = normalize_text(remove_prefixes(test_sample.answer_text))
        if prr < 1:
            wrong_answers.append(
                {
                    "answer": [normalize_text(remove_prefixes(a)) for a in pred_answers],
                    "question": test_sample.question,
                    "correct_answer": [normalize_text(remove_prefixes(a["text"])) for a in test_sample.all_answers],
                    "pRR": round(prr, 5),
                    "type": find_interrogatives(test_sample.question),
                    "context": test_sample.context,
                }
            )

wrong_answers = sorted(wrong_answers, key=lambda d: d["type"])

for k, v in groupby(wrong_answers, key=lambda a: a["type"]):
    typed_wrong_answers = sorted(list(v), key=lambda a: a["pRR"])
    print(40 * "*")
    print(k, len(typed_wrong_answers))
    print(40 * "*")
    print(
        round(np.mean([a["pRR"] for a in typed_wrong_answers]), 2),
        "±",
        round(np.std([a["pRR"] for a in typed_wrong_answers]), 2),
        f"Loss in pRR = {len(typed_wrong_answers) - sum([a['pRR'] for a in typed_wrong_answers])}",
    )

if False and input("print questions? [y]/n: ") != "n":
    for k, v in groupby(wrong_answers, key=lambda a: a["type"]):
        typed_wrong_answers = sorted(list(v), key=lambda a: a["pRR"])
        print(40 * "*")
        print(k, len(typed_wrong_answers))
        print(40 * "*")
        for wrong_answer in typed_wrong_answers:
            print("pRR:", wrong_answer["pRR"])
            print("Q:", wrong_answer["question"])
            print("C:", wrong_answer["context"])
            print("L:", wrong_answer["correct_answer"])
            print("A:", wrong_answer["answer"])
            print(wrong_answer["type"])
            print()

with open("data/smash_run01.json", "w") as f:
    submission = {
        id: [
            {"answer": answer, "rank": i+1, "score": 0.2}
            for i, answer in enumerate(answers_list)
        ]
        for id, answers_list in zip(ids, answers)
    }
    json.dump(submission, f, ensure_ascii=False)
