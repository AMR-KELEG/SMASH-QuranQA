import requests

import re
import string

import sys
from tqdm import tqdm
from colorama import Fore

import numpy as np
from settings import MAX_SEQ_LENGTH
from torch import nn


def get_ner_labels(sentence, tokens, ner_char_ranges):
    is_char_in_range = [False for _ in range(len(sentence))]
    for st, en in ner_char_ranges:
        for index in range(st, en):
            is_char_in_range[index] = True

    ner_labels = [
        1 if sum(is_char_in_range[start:end]) > 0 else 0
        for start, end in tokens.offsets
    ]

    return ner_labels


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

    def preprocess(self, tokenizer, use_multiple_answers=False):
        if not tokenizer:
            return None

        context = self.context
        question = self.question

        # Tokenize the strings into subwords
        tokenized_context = tokenizer.encode(context)
        tokenized_question = tokenizer.encode(question)

        persons_mentions_context = get_persons(context)
        persons_mentions_question = get_persons(question)
        # TODO: Handle multiple answers
        if use_multiple_answers and not self.all_answers:
            pass

        # Only use the first answer
        elif self.answer_text:
            answer = self.answer_text
            # Find the end character
            end_char_idx = self.start_char_idx + len(answer)

            is_char_in_ans = [0] * len(context)
            for idx in range(self.start_char_idx, end_char_idx):
                is_char_in_ans[idx] = 1

            # Find subtokens having any overlap with the range of characters
            ans_token_idx = []
            for idx, (start, end) in enumerate(tokenized_context.offsets):
                if sum(is_char_in_ans[start:end]) > 0:
                    ans_token_idx.append(idx)

            # Specify the indecies of the start and end subtokens
            self.start_token_idx = ans_token_idx[0]
            self.end_token_idx = ans_token_idx[-1]

            self.ner_labels = (
                get_ner_labels(context, tokenized_context, persons_mentions_context)
                + get_ner_labels(
                    question, tokenized_question, persons_mentions_question
                )[1:]
            )

        # Keeps ids of context and drop [CLS] from question
        # Both has [SEP] at the end which is desired
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        # Form the segment ids
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        # Attend to all these tokens
        attention_mask = [1] * len(input_ids)

        # Form padding
        padding_length = MAX_SEQ_LENGTH - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            # Avoid attending to paddings
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            if self.ner_labels:
                # -100 is the CrossEntropyLoss ignore_index
                self.ner_labels = self.ner_labels + ([-100] * padding_length)

        elif padding_length < 0:
            # TODO: Do some logging
            input_ids = input_ids[:MAX_SEQ_LENGTH]
            attention_mask = attention_mask[:MAX_SEQ_LENGTH]
            token_type_ids = token_type_ids[:MAX_SEQ_LENGTH]
            if self.ner_labels:
                self.ner_labels = self.ner_labels[:MAX_SEQ_LENGTH]

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
            all_answers = line["answers"]
            answer_text = all_answers[0]["text"]
            start_char_idx = all_answers[0]["start_char"]
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
        "ner_labels": [],
    }
    # Form list of values from the Sample objects
    for item in squad_examples:
        for key in dataset_dict:
            dataset_dict[key].append(getattr(item, key))

    # Transform the lists into numpy arrays
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    # Form the x, y tensors
    x = [
        dataset_dict["input_word_ids"],
        dataset_dict["input_mask"],
        dataset_dict["input_type_ids"],
    ]
    y = [
        dataset_dict["start_token_idx"],
        dataset_dict["end_token_idx"],
        dataset_dict["ner_labels"],
    ]
    return x, y


def normalize_text(text):
    # TODO: Handle Arabic text
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return text


def download_dataset():
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


def get_persons(passage):
    # Handle cases of person having proclitics OR
    # having Alef+Tanween as enclitic
    with open("data/persons.txt", "r") as f:
        PERSONS = [l.strip() for l in f]

    persons_mentions_ranges = []
    for PERSON in PERSONS:
        regexps = [rf"{PERSON}\b", rf"{PERSON}ا\b"]
        for regexp in regexps:
            for match in re.finditer(regexp, passage):
                persons_mentions_ranges.append((match.start(), match.end()))

    mentions = sorted(persons_mentions_ranges)

    if not mentions:
        return []

    def intersects(m1, m2):
        #  Make sure start1 < start2
        m1, m2 = sorted((m1, m2))

        start1, end1 = m1
        start2, end2 = m2
        if start2 >= start1 and start2 <= end1:
            return True
        return False

    def merge(m1, m2):
        m1, m2 = sorted((m1, m2))
        return (m1[0], m2[1])

    ranges = [mentions[0]]
    for cur_range in mentions[1:]:
        if intersects(ranges[-1], cur_range):
            ranges[-1] = merge(ranges[-1], cur_range)
        else:
            ranges.append(cur_range)
    return ranges
