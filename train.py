import re
import os
import sys
import json
import math
import argparse

import numpy as np
from tqdm import tqdm
from colorama import Fore
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from data_utils import (
    create_squad_examples,
    create_inputs_targets,
    normalize_text,
    download_dataset,
    load_dataset_as_tensors,
)
from settings import (
    GPU_ID,
    EPOCHS,
    BATCH_SIZE,
    CROSS_ENTROPY_IGNORE_INDEX,
    MODEL_NAME,
)
import logging
from model import MultiTaskQAModel

# TODO: Fix this!
sys.path.append("../")
sys.path.append("../quranqa/code/")

from quranqa.code.quranqa22_eval import (
    pRR_max_over_ground_truths,
)

from generate_new_splits import main as generate_faithful_splits
from glob import glob
from utils import mask_passage


def evaluate_model(
    eval_squad_examples, validation_data_loader, eval_step, dataset_name
):
    # Global variables: model,
    # Run validation
    validation_pbar = tqdm(
        total=len(eval_squad_examples),
        position=0,
        leave=True,
        file=sys.stdout,
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
    )
    model.eval()
    # TODO: Fix this!
    currentIdx = 0

    count = 0
    all_pRRs = []
    for batch in validation_data_loader:
        batch = tuple(t.to(GPU_ID) for t in batch)
        (
            input_word_ids,
            input_mask,
            input_type_ids,
            question_ids,
            start_token_idx,
            end_token_idx,
            ner_labels,
        ) = batch
        with torch.no_grad():
            outputs = model(
                input_ids=input_word_ids,
                attention_mask=input_mask,
                token_type_ids=input_type_ids,
                ner_labels=ner_labels,
                question_ids=question_ids,
            )

            start_logits, end_logits = outputs[1].split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=CROSS_ENTROPY_IGNORE_INDEX)
            start_loss = loss_fct(start_logits, start_token_idx)
            end_loss = loss_fct(end_logits, end_token_idx)
            total_loss = (start_loss + end_loss) / 2
            writer.add_scalar(
                f"Evaluation loss ({dataset_name}) /QA", total_loss.item(), eval_step
            )
            if USE_MULTITASK:
                ner_loss_fct = nn.CrossEntropyLoss(
                    ignore_index=CROSS_ENTROPY_IGNORE_INDEX
                )
                ner_loss = ner_loss_fct(outputs[2].view(-1, 2), ner_labels.view(-1))
                writer.add_scalar(
                    f"Evaluation loss ({dataset_name}) /NER", ner_loss.item(), eval_step
                )

            pred_start, pred_end = (
                start_logits.detach().cpu().numpy(),
                end_logits.detach().cpu().numpy(),
            )

        # TODO: Refactor this!
        # Compute evaluation metrics (e.g.: EM)
        pRRs = []
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            qa_sample = eval_squad_examples[currentIdx]
            currentIdx += 1
            offsets = qa_sample.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                # TODO: There is a problem here?!
                pRRs.append(0)
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = qa_sample.context[pred_char_start:pred_char_end]
            else:
                pred_ans = qa_sample.context[pred_char_start:]

            # TODO: Compute the right metrics
            pRR = pRR_max_over_ground_truths(
                [pred_ans],
                [[0]],
                [{"text": a["text"]} for a in qa_sample.all_answers],
            )
            pRRs.append(pRR)

            normalized_pred_ans = normalize_text(pred_ans)
            normalized_true_ans = [
                normalize_text(_["text"]) for _ in qa_sample.all_answers
            ]
            # Change the EM to use the repo's code!
            if normalized_pred_ans in normalized_true_ans:
                count += 1
        all_pRRs += pRRs
        validation_pbar.update(input_word_ids.size(0))
    acc = count / len(eval_squad_examples)
    validation_pbar.close()
    avg_pRR = sum(all_pRRs) / len(all_pRRs)
    writer.add_scalar(f"Evaluation pRR ({dataset_name})", avg_pRR, eval_step)
    return {"EM": acc, "pRR": avg_pRR}


def load_eval_dataset(filename):
    eval_data = load_dataset_as_tensors(
        filename, "Creating eval points", tokenizer, args.question_first
    )
    with open(filename, "r") as f:
        raw_eval_data = [json.loads(l) for l in f]
    eval_squad_examples = create_squad_examples(
        raw_eval_data, "", tokenizer, args.question_first
    )
    print(f"{len(eval_data)} evaluation points created.")
    eval_sampler = SequentialSampler(eval_data)
    validation_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
    return eval_squad_examples, validation_data_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT on QA.")
    parser.add_argument(
        "--ner",
        default=False,
        action="store_true",
        help="Use the model fine-tuned for NER.",
    )
    parser.add_argument(
        "--use_TAPT",
        default=False,
        action="store_true",
        help="Use the model further PT on quran.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        help="The value of the random seed to use.",
    )
    parser.add_argument(
        "--dropout_p",
        default=0,
        help="The value of the dropout probability after BERT.",
    )
    parser.add_argument(
        "--question_first",
        default=False,
        action="store_true",
        help="Use question as segment A, and passage as segment B",
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
        "--use_ner_multitasking",
        default=False,
        action="store_true",
        help="Use NER as an auxillary task for the model to learn.",
    )
    parser.add_argument(
        "--use_masking",
        default=False,
        action="store_true",
        help="Mask part of the input to avoid overfitting.",
    )
    parser.add_argument("--desc", required=True, help="The description of the model.")
    args = parser.parse_args()
    USE_MULTITASK = args.use_ner_multitasking

    # Try setting the seed
    torch.manual_seed(args.seed)

    writer = SummaryWriter(log_dir="training_logs")
    # writer.add_scalar('Loss/train', np.random.random(), global_step=n_iter)

    model_name = MODEL_NAME
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

    # Generate faithful splits
    generate_faithful_splits()

    # Make the tokenization faster
    slow_tokenizer = BertTokenizer.from_pretrained(model_name)
    # TODO: Rename the directory
    if not os.path.exists(f"{model_name}_/"):
        os.makedirs(f"{model_name}_/")
    slow_tokenizer.save_pretrained(f"{model_name}_/")
    tokenizer = BertWordPieceTokenizer(f"{model_name}_/vocab.txt", lowercase=True)

    # Organize the dataset into tensors
    train_data = load_dataset_as_tensors(
        "data/faithful_train.jsonl",
        "Creating training points",
        tokenizer,
        args.question_first,
    )

    eval_dataset_filenames = sorted(glob("data/*_eval.jsonl"))
    eval_datasets = []
    train_datasets = []
    for dataset_filename in eval_dataset_filenames:
        dataset_name = re.sub(r"_eval[.]jsonl", "", dataset_filename.split("/")[-1])
        eval_squad_examples, validation_data_loader = load_eval_dataset(
            dataset_filename
        )
        eval_datasets.append(
            {
                "name": dataset_name,
                "samples": eval_squad_examples,
                "dataloader": validation_data_loader,
            }
        )

    dataset_name = "eval"
    dataset_filename = "data/faithful_eval.jsonl "
    eval_squad_examples, validation_data_loader = load_eval_dataset(dataset_filename)
    eval_datasets.append(
        {
            "name": dataset_name,
            "samples": eval_squad_examples,
            "dataloader": validation_data_loader,
        }
    )

    train_squad_examples, train_data_loader = load_eval_dataset(
        "data/faithful_train.jsonl"
    )
    train_datasets.append(
        {
            "name": "train",
            "samples": train_squad_examples,
            "dataloader": train_data_loader,
        }
    )
    print(f"{len(train_data)} training points created.")
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(
        train_data, sampler=train_sampler, batch_size=BATCH_SIZE
    )

    # TODO: Continue pretraining
    # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForPreTraining
    model = MultiTaskQAModel(
        model_name,
        dropout_p=args.dropout_p,
        use_TAPT=args.use_TAPT,
        embed_ner=args.embed_ner,
        embed_question=args.embed_question,
    )
    model = model.to(device=GPU_ID)

    # Prepare optimizer
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

    ner_loss = nn.CrossEntropyLoss()

    cur_step = 0
    eval_steps = {dataset["name"]: 0 for dataset in eval_datasets}
    eval_steps["train"] = 0

    # Make masking percentage go to (1/512)
    initial_masking_percentage = math.exp(math.log(1 / 512) / 16)
    for epoch in range(1, EPOCHS + 1):
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
        print(f"Epoch {epoch}, masking percentage {initial_masking_percentage:.6f}")

        # Train the model using batches of data
        for step, batch in enumerate(train_data_loader):
            batch = tuple(t.to(GPU_ID) for t in batch)
            (
                input_word_ids,
                input_mask,
                input_type_ids,
                question_ids,
                start_token_idx,
                end_token_idx,
                ner_labels,
            ) = batch
            # Mask some of the input
            if args.use_masking:
                if epoch >= EPOCHS - 2:
                    masking_percentage = initial_masking_percentage / (epoch + 1)
                else:
                    masking_percentage = initial_masking_percentage ** (epochs + 1)
                input_ids = mask_passage(
                    input_word_ids, slow_tokenizer, masking_percentage
                )

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_word_ids,
                attention_mask=input_mask,
                token_type_ids=input_type_ids,
                ner_labels=ner_labels,
                question_ids=question_ids,
            )

            # Loss of QA
            start_logits, end_logits = outputs[1].split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            # TODO: How is this loss computed?
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_token_idx)
            end_loss = loss_fct(end_logits, end_token_idx)
            total_loss = (start_loss + end_loss) / (2 * BATCH_SIZE)

            # Don't use multitask learning
            if not USE_MULTITASK:
                loss = total_loss
            else:
                # Loss of NER
                ner_loss_fct = nn.CrossEntropyLoss(
                    ignore_index=CROSS_ENTROPY_IGNORE_INDEX
                )

                # TODO: This is a hyperparameter that needs tuning
                ALPHA = 0.99
                ner_loss = ner_loss_fct(outputs[2].view(-1, 2), ner_labels.view(-1))
                loss = ALPHA * total_loss + (1 - ALPHA) * ner_loss
                writer.add_scalar("Training loss/NER", ner_loss.item(), cur_step)

            writer.add_scalar("Training loss/QA", total_loss.item(), cur_step)
            cur_step += 1

            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            training_pbar.update(BATCH_SIZE)

        training_pbar.close()
        torch.save(
            model.state_dict(),
            f"checkpoints/weights_{args.desc}_seed_{args.seed}_" + str(epoch) + ".pth",
        )

        dataset_name = "train"
        eval_results = evaluate_model(
            train_datasets[0]["samples"],
            train_datasets[0]["dataloader"],
            eval_steps[dataset_name],
            dataset_name,
        )
        eval_steps[dataset_name] = eval_steps[dataset_name] + 1
        print(
            f"\nDataset: {dataset_name}, Epoch={epoch}, exact match score={eval_results['EM']:.2f}, "
            f"pRR={eval_results['pRR']:.2f}, training loss: {tr_loss}"
        )

        for evaluation_dataset_dict in eval_datasets:
            dataset_name = evaluation_dataset_dict["name"]
            eval_results = evaluate_model(
                evaluation_dataset_dict["samples"],
                evaluation_dataset_dict["dataloader"],
                eval_steps[dataset_name],
                dataset_name,
            )
            eval_steps[dataset_name] = eval_steps[dataset_name] + 1

            print(
                f"\nDataset: {dataset_name}, Epoch={epoch}, exact match score={eval_results['EM']:.2f}, "
                f"pRR={eval_results['pRR']:.2f}, training loss: {tr_loss}"
            )
    writer.close()
