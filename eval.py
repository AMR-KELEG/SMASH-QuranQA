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
from settings import GPU_ID, EPOCHS
from data_utils import (
    create_squad_examples,
    create_inputs_targets,
    load_dataset_as_tensors,
    load_samples_as_tensors,
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
parser.add_argument(
    "--train", action="store_true", help="Run evaluation on the training data."
)
parser.add_argument(
    "--test", action="store_true", help="Run evaluation on the testing data."
)
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
parser.add_argument(
    "--question_first",
    default=False,
    action="store_true",
    help="Use question as segment A, and passage as segment B",
)
parser.add_argument(
    "--use_TAPT",
    default=False,
    action="store_true",
    help="Use the model further PT on quran.",
)
parser.add_argument(
    "--embed_ner",
    default=False,
    action="store_true",
    help="Embed NERs as input to BERT layers.",
)
parser.add_argument(
    "--embed_question",
    default=False,
    action="store_true",
    help="Embed Question type as input to BERT layers.",
)
parser.add_argument(
    "--model_name",
    default="CAMeL-Lab/bert-base-arabic-camelbert-ca",
    help="The name of the BERT model to fine-tune.",
)
parser.add_argument(
    "--use_stemming",
    default=False,
    action="store_true",
    help="Stem the tokens before feeding them into the model.",
)
parser.add_argument(
    "--run_id",
    default="1",
    help="An integer for the run id.",
)

parser.add_argument("--desc", required=True, help="The description of the model.")
args = parser.parse_args()
MODEL_NAME = args.model_name

if args.test:
    datafile = "data/test.jsonl"
elif args.train:
    datafile = "data/train_ar.jsonl"
else:
    datafile = "data/eval_ar.jsonl"

# Load the tokenizer
tokenizer = BertWordPieceTokenizer(f"{MODEL_NAME}_/vocab.txt", lowercase=True)

# Load the data
with open(datafile, "r") as f:
    raw_eval_data = [json.loads(l) for l in f]

# Load the trained model
model = MultiTaskQAModel(
    MODEL_NAME,
    use_TAPT=args.use_TAPT,
    embed_ner=args.embed_ner,
    embed_question=args.embed_question,
).to(device=GPU_ID)
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
    (
        input_word_ids,
        input_mask,
        input_type_ids,
        question_ids,
        ner_labels,
    ) = load_samples_as_tensors(
        batch_data,
        "Loading sample",
        tokenizer,
        question_first=args.question_first,
        use_stemming=args.use_stemming,
    )
    outputs = model(
        input_ids=input_word_ids.to(GPU_ID),
        attention_mask=input_mask.to(GPU_ID),
        token_type_ids=input_type_ids.to(GPU_ID),
        question_ids=question_ids.to(GPU_ID),
        ner_labels=ner_labels.to(GPU_ID),
    )

    start_logits, end_logits = outputs[1].split(1, dim=-1)

    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)
    pred_start, pred_end = (
        start_logits.detach().cpu().numpy(),
        end_logits.detach().cpu().numpy(),
    )

    test_sample = create_squad_examples(
        batch_data,
        "Sample",
        tokenizer,
        question_first=args.question_first,
        use_stemming=args.use_stemming,
    )[0]
    try:
        offsets = test_sample.context_token_to_char
    except:
        # TODO: This is a hack added by Amr!
        # Investigate the reason for this!
        offsets = []

    pred_answers = []
    start = np.argmax(pred_start)
    end = np.argmax(pred_end)
    if start >= len(offsets):
        print("Fallback for start!")
        start = 0

    pred_char_start = offsets[start][0]
    if end < len(offsets) and end >= start:
        pred_ans = test_sample.context[pred_char_start : offsets[end][1]]
    else:
        pred_ans = test_sample.context[pred_char_start:]
    pred_answers.append(pred_ans)
    pred_answers.append(test_sample.context)

    # Find most probable spans!
    # spans = get_spans(pred_start, pred_end)
    # for p, start, end in spans:
    #     if start >= len(offsets):
    #         print("Fallback for start!")
    #         start = 0

    #     pred_char_start = offsets[start][0]
    #     if end < len(offsets) and end >= start:
    #         pred_ans = test_sample.context[pred_char_start : offsets[end][1]]
    #     else:
    #         pred_ans = test_sample.context[pred_char_start:]
    #     pred_answers.append(pred_ans)

    # Compute pRR stats if answers are there
    if test_sample.all_answers:
        prr = pRR_max_over_ground_truths(
            pred_answers,
            [[0] for _ in range(0, len(pred_answers))],
            [{"text": a["text"]} for a in test_sample.all_answers],
        )
        # TODO: Get rid of this part
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

    else:
        # Just a placeholder for values!
        prr = 0

    answers.append(pred_answers)
    ids.append(test_sample.pq_id)
    full_text_answers.append(test_sample.context)
    prrs.append(prr)

if wrong_answers:
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

with open(f"data/SMASH_run0{args.run_id}.json", "w") as f:
    submission = {
        id: [
            {"answer": a, "rank": i + 1, "score": round(1/len(answers_list), 2)}
            for i, a in enumerate(answers_list)
        ]
        for id, answers_list, full_answer in zip(ids, answers, full_text_answers)
    }
    json.dump(submission, f, ensure_ascii=False)

with open(
    f"data/debug_{'test' if args.test else 'train' if args.train else 'dev'}.jsonl", "w"
) as f:
    for s, a, prr in zip(raw_eval_data, answers, prrs):
        s["pred"] = a
        s["pRR"] = prr
        f.write(json.dumps(s, ensure_ascii=False))
        f.write("\n")
