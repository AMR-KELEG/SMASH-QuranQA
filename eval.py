import json
import os
import re
import sys

# TODO: Fix this!
sys.path.append("../")
sys.path.append("../quranqa/code/")

import numpy as np
from colorama import Fore
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler
from settings import GPU_ID, MODEL_NAME, EPOCHS
from data_utils import (
    create_squad_examples,
    create_inputs_targets,
    load_dataset_as_tensors,
)
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
    "--seed", default=0, help="The value of the random seed to use.",
)
parser.add_argument(
    "--epoch",
    default=EPOCHS,
    help="The value of the epoch at which the checkpoint was generated.",
)
parser.add_argument(
    "--use_TAPT",
    default=False,
    action="store_true",
    help="Use the model further PT on quran.",
)
parser.add_argument("--desc", required=True, help="The description of the model.")
args = parser.parse_args()

if args.train:
    datafile = "data/train_ar.jsonl"
else:
    datafile = "data/eval_ar.jsonl"

# Load the tokenizer
tokenizer = BertWordPieceTokenizer(f"{MODEL_NAME}_/vocab.txt", lowercase=True)

# Load the data
with open(datafile, "r") as f:
    raw_eval_data = [json.loads(l) for l in f]
eval_data = load_dataset_as_tensors(datafile, tokenizer, "Loading data")
print(f"{len(eval_data)} evaluation points created.")
eval_sampler = SequentialSampler(eval_data)
validation_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

# Load the trained model
model = MultiTaskQAModel(MODEL_NAME, use_TAPT=args.use_TAPT).to(device=GPU_ID)
model.load_state_dict(
    torch.load(f"checkpoints/weights_{args.desc}_seed_{args.seed}_{args.epoch}.pth")
)
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

        start = np.argmax(start)
        end = np.argmax(end)

        pred_char_start = offsets[start][0]
        if end < len(offsets):
            pred_ans = test_sample.context[pred_char_start : offsets[end][1]]
        else:
            pred_ans = test_sample.context[pred_char_start:]

        prr = pRR_max_over_ground_truths(
            [pred_ans, test_sample.context],
            [[pred_char_start], [0]],
            [{"text": a["text"]} for a in test_sample.all_answers],
        )

        answers.append(pred_ans)
        ids.append(test_sample.question_id)
        full_text_answers.append(test_sample.context)
        prrs.append(prr)

        cleaned_pred = normalize_text(remove_prefixes(pred_ans))
        cleaned_ans = normalize_text(remove_prefixes(test_sample.answer_text))
        if cleaned_ans != cleaned_pred:
            wrong_answers.append(
                {
                    "answer": cleaned_pred,
                    "question": test_sample.question,
                    "correct_answer": cleaned_ans,
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
            print("L:", wrong_answer["correct_answer"])
            print("A:", wrong_answer["answer"])
            print(wrong_answer["type"])
            print()

with open("data/smash_run01.json", "w") as f:
    submission = {
        id: [
            {"answer": answer, "rank": 1, "score": 0.99},
            {"answer": full_answer, "rank": 2, "score": 0.01},
        ]
        for id, answer, full_answer in zip(ids, answers, full_text_answers)
    }
    json.dump(submission, f, ensure_ascii=False)
