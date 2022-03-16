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


class Sample:
    def __init__(
        self,
        question,
        context,
        start_char_idx=None,
        answer_text=None,
        all_answers=None,
        question_id=None,
    ):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
        self.start_token_idx = -1
        self.end_token_idx = -1
        self.question_id = question_id

    def preprocess(self, tokenizer):
        if not tokenizer:
            return None

        context = " ".join(str(self.context).split())
        question = " ".join(str(self.question).split())

        tokenized_context = tokenizer.encode(context)
        tokenized_question = tokenizer.encode(question)
        if self.answer_text is not None:
            answer = " ".join(str(self.answer_text).split())
            end_char_idx = self.start_char_idx + len(answer)
            if end_char_idx >= len(context):
                self.skip = True
                return
            is_char_in_ans = [0] * len(context)
            for idx in range(self.start_char_idx, end_char_idx):
                is_char_in_ans[idx] = 1
            ans_token_idx = []
            for idx, (start, end) in enumerate(tokenized_context.offsets):
                if sum(is_char_in_ans[start:end]) > 0:
                    ans_token_idx.append(idx)
            if len(ans_token_idx) == 0:
                self.skip = True
                return
            self.start_token_idx = ans_token_idx[0]
            self.end_token_idx = ans_token_idx[-1]
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:
            self.skip = True
            return
        self.input_word_ids = input_ids
        self.input_type_ids = token_type_ids
        self.input_mask = attention_mask
        self.context_token_to_char = tokenized_context.offsets


def create_squad_examples(raw_data, desc, tokenizer):
    # TODO: Pass the tokenizer as a parameter for this function
    p_bar = tqdm(
        total=len(raw_data),
        desc=desc,
        position=0,
        leave=True,
        file=sys.stdout,
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
    )
    squad_examples = []
    for line in raw_data:
        question_id = line["pq_id"]
        context = line["passage"]
        question = line["question"]
        # TODO: Handle if answers aren't there
        if line["answers"]:
            all_answers = [a["text"] for a in line["answers"]]
            answer_text = all_answers[0]
            start_char_idx = line["answers"][0]["start_char"]
            squad_eg = Sample(
                question,
                context,
                start_char_idx,
                answer_text,
                all_answers,
                question_id=question_id,
            )
        else:
            squad_eg = Sample(question, context, question_id=question_id)
        squad_eg.preprocess(tokenizer)
        squad_examples.append(squad_eg)
        p_bar.update(1)
    p_bar.close()
    return squad_examples


def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_word_ids": [],
        "input_type_ids": [],
        "input_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip is False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])
    x = [
        dataset_dict["input_word_ids"],
        dataset_dict["input_mask"],
        dataset_dict["input_type_ids"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


def normalize_text(text):
    # TODO: Handle Arabic text
    # text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    # regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    # text = re.sub(regex, " ", text)
    # text = " ".join(text.split())
    return text
