import os
import re
import sys
import json
import argparse

import numpy as np
from tqdm import tqdm
from colorama import Fore
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from data_utils import (
    Sample,
    create_squad_examples,
    create_inputs_targets,
    normalize_text,
    download_dataset,
)
from settings import GPU_ID, EPOCHS, MAX_SEQ_LENGTH, BATCH_SIZE
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT on QA.")
    parser.add_argument(
        "--ner",
        default=False,
        action="store_true",
        help="Use the model fine-tuned for NER.",
    )
    parser.add_argument(
        "--seed", default=0, help="The value of the random seed to use.",
    )
    args = parser.parse_args()

    # Try setting the seed
    torch.manual_seed(args.seed)

    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
    # Use the model that was fine-tuned for NER
    if args.ner:
        model_name = model_name + "-ner"

    logger = logging.getLogger("Train")
    # Create directories
    for directory in ["data", "checkpoints"]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Download the data from the repository
    download_dataset()

    # Load the data into list of dictionaries
    with open("data/train_ar.jsonl") as f:
        raw_train_data = [json.loads(l) for l in f]
    with open("data/eval_ar.jsonl") as f:
        raw_eval_data = [json.loads(l) for l in f]

    # Make the tokenization faster
    slow_tokenizer = BertTokenizer.from_pretrained(model_name)
    if not os.path.exists(f"{model_name}_/"):
        os.makedirs(f"{model_name}_/")
    slow_tokenizer.save_pretrained(f"{model_name}_/")
    tokenizer = BertWordPieceTokenizer(f"{model_name}_/vocab.txt", lowercase=True)

    # Organize the dataset into tensors
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
    logger.info(f"{len(train_data)} training points created.")
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(
        train_data, sampler=train_sampler, batch_size=BATCH_SIZE
    )
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
        eval_data, sampler=eval_sampler, batch_size=BATCH_SIZE
    )

    # TODO: Continue pretraining
    # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForPreTraining
    model = BertForQuestionAnswering.from_pretrained(model_name)
    model.qa_outputs = nn.Linear(768, 2)
    model = model.to(device=GPU_ID)
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
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]

    optimizer = torch.optim.Adam(
        lr=1e-5, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters
    )

    for epoch in range(1, EPOCHS + 1):
        logger.info("Training epoch ", str(epoch))
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

        # Train the model using batches of data
        for step, batch in enumerate(train_data_loader):
            batch = tuple(t.to(GPU_ID) for t in batch)
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
        logger.info(f"\nTraining loss={tr_loss / nb_tr_steps:.4f}")
        torch.save(model.state_dict(), "checkpoints/weights_" + str(epoch) + ".pth")

        # Run validation
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
            batch = tuple(t.to(GPU_ID) for t in batch)
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
        logger.info(f"\nEpoch={epoch}, exact match score={acc:.2f}")
