import requests

import re
import string

import sys
import json
from tqdm import tqdm
from colorama import Fore

import numpy as np
from settings import MAX_SEQ_LENGTH, CROSS_ENTROPY_IGNORE_INDEX
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from utils import find_interrogative_index


from farasa.stemmer import FarasaStemmer

STEMMER = FarasaStemmer(interactive=True)


def get_ner_labels(sentence, tokens, ner_char_ranges):
    """
    Assign a lable of 0 (not a named-entity), 1 (named-entity) to tokens
    """

    char_is_in_an_ner_range = [False for _ in range(len(sentence))]
    for st, en in ner_char_ranges:
        for index in range(st, en):
            char_is_in_an_ner_range[index] = True

    # Assign the token a label of 1 if any of its characters is within an ner range
    ner_labels = [
        1 if sum(char_is_in_an_ner_range[start:end]) > 0 else 0
        for start, end in tokens.offsets
    ]

    return ner_labels


def stem_tokenize(s):
    original_offsets, stemmed_offsets = [], []
    stemmed_tokens = []
    cur_token_start = 0
    cur_length = -1
    for i, c in enumerate(s):
        if c in [" ", ".", "؟", '"']:
            token = s[cur_token_start:i]
            if token.strip():
                stemmed_token = STEMMER.stem(token)
                stemmed_tokens.append(stemmed_token)
                original_offsets.append((cur_token_start, i))
                stemmed_offsets.append(
                    (1 + cur_length, 1 + cur_length + len(stemmed_token))
                )
                cur_length += len(stemmed_token) + 1
            cur_token_start = i + 1

        if c in [".", "؟", '"']:
            token = c
            stemmed_token = STEMMER.stem(token)
            stemmed_tokens.append(stemmed_token)
            original_offsets.append((i, i + 1))
            stemmed_offsets.append(
                (1 + cur_length, 1 + cur_length + len(stemmed_token))
            )
            cur_length += 2

    token = s[cur_token_start:i]
    if token.strip():
        # Make sure the token isn't just a space
        stemmed_token = STEMMER.stem(token)
        stemmed_tokens.append(stemmed_token)
        original_offsets.append((cur_token_start, i))
        stemmed_offsets.append((1 + cur_length, 1 + cur_length + len(stemmed_token)))
    return " ".join(stemmed_tokens), original_offsets, stemmed_offsets


class Sample:
    def __init__(
        self,
        question,
        context,
        start_char_idx=None,
        answer_text=None,
        all_answers=None,
        pq_id=None,
    ):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
        self.start_token_idx = -1
        self.end_token_idx = -1
        self.pq_id = pq_id

    def preprocess(self, tokenizer, question_first=False, use_stemming=False):
        if not tokenizer:
            return None

        context = re.sub(r"\s+", " ", self.context)
        question = re.sub(r"\s+", " ", self.question)

        if use_stemming:
            context, original_offsets, stemmed_offsets = stem_tokenize(context)
            question, _, _ = stem_tokenize(question)
            # TODO: Store the offsets in order to decode the answer later
            self.original_offsets = original_offsets
            self.stemmed_offsets = stemmed_offsets

        # Tokenize the strings into subwords
        tokenized_context = tokenizer.encode(context)
        tokenized_question = tokenizer.encode(question)

        if not use_stemming:
            # Use the same offsets from the main tokenizer
            # Ignore the [CLS] and [SEP] tokens
            self.original_offsets = tokenized_context.offsets[1:-1]
            self.stemmed_offsets = self.original_offsets

        persons_mentions_context = get_persons(context)
        persons_mentions_question = get_persons(question)

        # Only use the first answer
        if self.answer_text:
            answer = self.answer_text
            # Find the end character
            start_char_idx = self.start_char_idx
            end_char_idx = self.start_char_idx + len(answer)

            mapped_start_idx, mapped_end_idx = -1, -1
            # These are old offsets
            for (orig_start, orig_end), (stemmed_start, stemmed_end) in zip(
                self.original_offsets, self.stemmed_offsets
            ):
                if start_char_idx >= orig_start and start_char_idx <= orig_end:
                    # use the stemmed offset
                    mapped_start_idx = stemmed_start
                if end_char_idx >= orig_start and end_char_idx <= orig_end:
                    # use the stemmed offset
                    mapped_end_idx = stemmed_end

            is_char_in_ans = [0] * len(context)
            for idx in range(mapped_start_idx, mapped_end_idx):
                is_char_in_ans[idx] = 1

            # Find subtokens having any overlap with the range of characters
            ans_token_idx = []
            for idx, (start, end) in enumerate(tokenized_context.offsets):
                # TODO: Do I need to skip first two indecies?
                # if idx == 0 or idx == len(tokenized_context.offsets):
                #     continue
                if sum(is_char_in_ans[start:end]) > 0:
                    ans_token_idx.append(idx)

            # Specify the indecies of the start and end subtokens
            self.start_token_idx = ans_token_idx[0]
            self.end_token_idx = ans_token_idx[-1]

        # Assign ner labels to the tokens in both the context and the question
        context_ner_labels = get_ner_labels(
            context, tokenized_context, persons_mentions_context
        )
        question_ner_labels = get_ner_labels(
            question, tokenized_question, persons_mentions_question
        )

        # TODO: Does the order matter? (i.e. Context, Question or Question, Context)
        if question_first:
            # Keeps ids of context and drop [CLS] from question
            # Both has [SEP] at the end which is desired
            input_ids = tokenized_question.ids + tokenized_context.ids[1:]
            # Form the segment ids
            token_type_ids = [0] * len(tokenized_question.ids) + [1] * len(
                tokenized_context.ids[1:]
            )
            ner_labels = question_ner_labels + context_ner_labels[1:]
            # Attend to all these tokens
            attention_mask = [1] * len(input_ids)
        else:
            # Keeps ids of context and drop [CLS] from question
            # Both has [SEP] at the end which is desired
            input_ids = tokenized_context.ids + tokenized_question.ids[1:]
            # Form the segment ids
            token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
                tokenized_question.ids[1:]
            )
            ner_labels = context_ner_labels + question_ner_labels[1:]
            # Attend to all these tokens
            attention_mask = [1] * len(input_ids)

        # Form padding
        padding_length = MAX_SEQ_LENGTH - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            # Avoid attending to paddings
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            if ner_labels:
                ner_labels = ner_labels + (
                    [CROSS_ENTROPY_IGNORE_INDEX] * padding_length
                )

        elif padding_length < 0:
            # TODO: Do some logging
            input_ids = input_ids[:MAX_SEQ_LENGTH]
            attention_mask = attention_mask[:MAX_SEQ_LENGTH]
            token_type_ids = token_type_ids[:MAX_SEQ_LENGTH]
            if ner_labels:
                ner_labels = ner_labels[:MAX_SEQ_LENGTH]

        self.input_word_ids = input_ids
        self.input_type_ids = token_type_ids
        self.input_mask = attention_mask
        self.context_token_to_char = tokenized_context.offsets
        self.ner_labels = ner_labels
        self.question_id = find_interrogative_index(question)


def create_squad_examples(
    raw_data, desc, tokenizer, question_first=False, use_stemming=False
):
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
        pq_id = line["pq_id"]
        context = line["passage"]
        question = line["question"]
        # TODO: Handle if answers aren't there
        if not line["answers"]:
            squad_eg = Sample(question, context, pq_id=pq_id)
            squad_eg.preprocess(
                tokenizer, question_first=question_first, use_stemming=use_stemming
            )
            squad_examples.append(squad_eg)
        else:
            # Use different valid answers as new samples
            for cur_answer in line["answers"]:
                all_answers = line["answers"]
                answer_text = cur_answer["text"]
                start_char_idx = cur_answer["start_char"]
                squad_eg = Sample(
                    question,
                    context,
                    start_char_idx,
                    answer_text,
                    all_answers,
                    pq_id=pq_id,
                )
                squad_eg.preprocess(
                    tokenizer, question_first=question_first, use_stemming=use_stemming
                )
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
        "question_id": [],
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
        dataset_dict["question_id"],
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
    test_data_file = "https://gitlab.com/bigirqu/quranqa/-/raw/main/datasets/qrcd_v1.1_test_noAnswers.jsonl"
    train_data = requests.get(train_data_file)
    if train_data.status_code in (200,):
        with open("data/train_ar.jsonl", "wb") as train_file:
            train_file.write(train_data.content)
    eval_data = requests.get(dev_data_file)
    if eval_data.status_code in (200,):
        with open("data/eval_ar.jsonl", "wb") as eval_file:
            eval_file.write(eval_data.content)
    test_data = requests.get(test_data_file)
    if test_data.status_code in (200,):
        with open("data/test_ar.jsonl", "wb") as test_file:
            test_file.write(test_data.content)


def get_persons(passage):
    # Handle cases of person having proclitics OR
    # having Alef+Tanween as enclitic
    with open("data/persons.txt", "r") as f:
        PERSONS = [l.strip() for l in f]

    with open("data/animals.txt", "r") as f:
        PERSONS += [l.strip() for l in f]

    persons_mentions_ranges = []
    for PERSON in PERSONS:
        regexps = [rf"{PERSON}\b", rf"{PERSON}ا\b"]
        for regexp in regexps:
            for match in re.finditer(regexp, passage):
                persons_mentions_ranges.append((match.start(), match.end()))

    mentions = sorted(persons_mentions_ranges)

    # Find mentions in the form اسم موصول للجمع + next word
    regexp = r"([ال](?:لذين|لا[تئ]ي) \S{4,})\b"
    if re.search(regexp, passage):
        mentions += [(m.start(), m.end()) for m in re.finditer(regexp, passage)]

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


def load_dataset_as_tensors(
    datafile, desc, tokenizer, question_first=False, use_stemming=False
):
    with open(datafile, "r") as f:
        raw_data = [json.loads(l) for l in f]
    squad_examples = create_squad_examples(
        raw_data,
        desc,
        tokenizer,
        question_first=question_first,
        use_stemming=use_stemming,
    )
    X, y = create_inputs_targets(squad_examples)
    tensor_data = TensorDataset(
        torch.tensor(X[0], dtype=torch.int64),
        torch.tensor(X[1], dtype=torch.float),
        torch.tensor(X[2], dtype=torch.int64),
        torch.tensor(X[3], dtype=torch.int64),
        torch.tensor(y[0], dtype=torch.int64),
        torch.tensor(y[1], dtype=torch.int64),
        torch.tensor(y[2], dtype=torch.int64),
    )
    return tensor_data


def load_samples_as_tensors(
    raw_data, desc, tokenizer, question_first=False, use_stemming=False
):
    squad_examples = create_squad_examples(
        raw_data,
        desc,
        tokenizer,
        question_first=question_first,
        use_stemming=use_stemming,
    )
    X, y = create_inputs_targets(squad_examples)

    return (
        torch.tensor(X[0], dtype=torch.int64),
        torch.tensor(X[1], dtype=torch.float),
        torch.tensor(X[2], dtype=torch.int64),
        torch.tensor(X[3], dtype=torch.int64),
        torch.tensor(y[2], dtype=torch.int64),
    )


def load_eval_dataset(filename, tokenizer, question_first, use_stemming):
    eval_data = load_dataset_as_tensors(
        filename,
        "Creating eval points",
        tokenizer,
        question_first=question_first,
        use_stemming=use_stemming,
    )
    with open(filename, "r") as f:
        raw_eval_data = [json.loads(l) for l in f]
    eval_squad_examples = create_squad_examples(
        raw_eval_data,
        "",
        tokenizer,
        question_first=question_first,
        use_stemming=use_stemming,
    )
    print(f"{len(eval_data)} evaluation points created.")
    eval_sampler = SequentialSampler(eval_data)
    validation_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
    return eval_squad_examples, validation_data_loader
