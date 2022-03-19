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
from settings import GPU_ID
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
args = parser.parse_args()

if args.train:
    datafile = "data/train_ar.jsonl"
else:
    datafile = "data/eval_ar.jsonl"

with open(datafile, "r") as f:
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
logger.info(f"{len(eval_data)} evaluation points created.")
eval_sampler = SequentialSampler(eval_data)
validation_data_loader = DataLoader(
    eval_data, sampler=eval_sampler, batch_size=1
)

# ============================================ TESTING =================================================================
model = MultiTaskQAModel(model_name).to(device=GPU_ID)
model.load_state_dict(torch.load("checkpoints/weights_16.pth"))
model.eval()

answers = []
full_text_answers = []
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
        test_sample = test_samples[idx]
        try:
            offsets = test_sample.context_token_to_char
        except:
            # TODO: This is a hack added by Amr!
            # Investigate the reason for this!
            offsets = []

        # TODO: Complete these spans!
        get_spans(start, end)

        start = np.argmax(start)
        end = np.argmax(end)

        pred_ans = None
        if start >= len(offsets):
            pred_ans = test_sample.context
        else:
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_ans = test_sample.context[pred_char_start : offsets[end][1]]
            else:
                pred_ans = test_sample.context[pred_char_start:]
        if not pred_ans:
            pred_ans = test_sample.context
        prr = pRR_max_over_ground_truths(
            [pred_ans], [[pred_char_start]], [{"text": test_sample.answer_text}]
        )
        answers.append(pred_ans)
        ids.append(test_sample.question_id)
        full_text_answers.append(test_sample.answer_text)
        prrs.append(prr)

        if normalize_text(remove_prefixes(pred_ans)) != normalize_text(
            remove_prefixes(test_sample.answer_text)
        ):
            wrong_answers.append(
                {
                    "answer": pred_ans,
                    "question": test_sample.question,
                    "correct_answer": test_sample.answer_text,
                    "pRR": round(prr, 5),
                    "type": find_interrogatives(test_sample.question),
                }
            )

wrong_answers = sorted(wrong_answers, key=lambda d: d["type"])

for k, v in groupby(wrong_answers, key=lambda a: a["type"]):
    typed_wrong_answers = sorted(list(v), key=lambda a: a["pRR"])
    logger.info(40 * "*")
    logger.info(k, len(typed_wrong_answers))
    logger.info(40 * "*")
    logger.info(
        round(np.mean([a["pRR"] for a in typed_wrong_answers]), 2),
        "±",
        round(np.std([a["pRR"] for a in typed_wrong_answers]), 2),
    )

if False and input("logger.info questions? [y]/n: ") != "n":
    for k, v in groupby(wrong_answers, key=lambda a: a["type"]):
        typed_wrong_answers = sorted(list(v), key=lambda a: a["pRR"])
        logger.info(40 * "*")
        logger.info(k, len(typed_wrong_answers))
        logger.info(40 * "*")
        for wrong_answer in typed_wrong_answers:
            logger.info("pRR:", wrong_answer["pRR"])
            logger.info("Q:", wrong_answer["question"])
            logger.info("L:", wrong_answer["correct_answer"])
            logger.info("A:", wrong_answer["answer"])
            logger.info(wrong_answer["type"])
            logger.info()

with open("data/smash_run01.json", "w") as f:
    submission = {
        id: [
            {"answer": answer, "rank": 1, "score": 0.99},
            {"answer": full_answer, "rank": 2, "score": 0.01},
        ]
        for id, answer, full_answer in zip(ids, answers, full_text_answers)
    }
    json.dump(submission, f, ensure_ascii=False)
