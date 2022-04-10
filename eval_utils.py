import sys
import torch
from tqdm import tqdm
from settings import *
import numpy as np
from data_utils import (
    create_squad_examples,
    create_inputs_targets,
    normalize_text,
    download_dataset,
    load_dataset_as_tensors,
    load_eval_dataset,
)
from colorama import Fore
import torch.nn as nn
from settings import (
    GPU_ID,
    EPOCHS,
    BATCH_SIZE,
    CROSS_ENTROPY_IGNORE_INDEX,
)
import logging
from model import MultiTaskQAModel
from settings import *

from quranqa22_eval import (
    pRR_max_over_ground_truths,
)


def evaluate_model(
    model,
    eval_squad_examples,
    validation_data_loader,
    eval_step,
    dataset_name,
    args,
    writer=None,
):
    USE_MULTITASK = args.use_ner_multitasking
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
            if writer:
                writer.add_scalar(
                    f"Evaluation loss ({dataset_name}) /QA",
                    total_loss.item(),
                    eval_step,
                )
            if USE_MULTITASK:
                ner_loss_fct = nn.CrossEntropyLoss(
                    ignore_index=CROSS_ENTROPY_IGNORE_INDEX
                )
                ner_loss = ner_loss_fct(outputs[2].view(-1, 2), ner_labels.view(-1))
                if writer:
                    writer.add_scalar(
                        f"Evaluation loss ({dataset_name}) /NER",
                        ner_loss.item(),
                        eval_step,
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

            stemmed_offsets = qa_sample.stemmed_offsets
            original_offsets = qa_sample.original_offsets
            subword_offsets = qa_sample.context_token_to_char

            # TODO: Fix this for question first mode!
            start = np.argmax(start)
            end = np.argmax(end)

            # Ignore first two indecies!
            if start >= len(subword_offsets):
                start = 1
            if end >= len(subword_offsets) - 1:
                end = len(subword_offsets) - 2

            pred_char_start = subword_offsets[start][0]
            pred_char_end = subword_offsets[end][1]

            stem_start_idx, stem_end_idx = -1, -1
            for i, (st, en) in enumerate(stemmed_offsets):
                if pred_char_start >= st and pred_char_start <= en:
                    stem_start_idx = i
                if pred_char_end >= st and pred_char_end <= en:
                    stem_end_idx = i

            pred_ans = qa_sample.context[
                original_offsets[stem_start_idx][0] : original_offsets[stem_end_idx][1]
            ]

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
    if writer:
        writer.add_scalar(f"Evaluation pRR ({dataset_name})", avg_pRR, eval_step)
    return {"EM": acc, "pRR": avg_pRR}
