import torch
import numpy as np


def softmax(x, temperature=1):
    x = x.reshape(-1)
    exp_x = np.exp(x / temperature)
    print(sum([round(v, 3) for v in exp_x / exp_x.sum()]))
    return [round(v, 3) for v in exp_x / exp_x.sum()]


INTERROGATIVE_ARTICLES = (
    "لماذا كيف بأي بماذا من كم كيف ما ماذا هل اين متى".split()
    + ["NA"]
    # + [
    #     "من هو",
    #     "من الذي",
    # ]
    # + ["ما المخلوقات", "ما هي انواع", "ما هي اسماء"]
)


def find_interrogatives(question):
    interrogatives = []
    indecies = []
    for article in INTERROGATIVE_ARTICLES:
        if article in question:
            interrogatives.append(article)
            indecies.append(question.index(article))

    # A hack to handle "من" as a preposition
    if len(set(interrogatives)) > 1 and "من" in interrogatives:
        interrogatives = [i for i in interrogatives if i != "من"]
        indecies = [i for i, inter in zip(indecies, interrogatives) if inter != "من"]

    final_interrogatives = []
    interrogatives = sorted(
        set([(index, inter) for inter, index in zip(interrogatives, indecies)])
    )
    for index, article in interrogatives:
        if any(
            [
                article in other_article and article != other_article
                for (other_index, other_article) in interrogatives
            ]
        ):
            continue
        final_interrogatives.append(article)

    return final_interrogatives[0] if final_interrogatives else "NA"


def find_interrogative_index(question):
    interrogative = find_interrogatives(question)
    return INTERROGATIVE_ARTICLES.index(interrogative)


def get_spans(start_logits, end_logits, temperature):
    # TODO: Use temperature!
    start_probs = softmax(start_logits.reshape(-1), temperature=temperature)
    end_probs = softmax(end_logits.reshape(-1), temperature=temperature)

    start_probs = sorted(
        [(p, i) for i, p in enumerate(start_probs)], key=lambda t: -t[0]
    )

    # Find best end
    span_probabilities = []

    for start_prob, start_index in start_probs[:5]:
        best_prob = 0
        best_end_index = start_index
        for end_index in range(start_index, len(end_probs)):
            if start_prob * end_probs[end_index] > best_prob:
                best_prob = start_prob * end_probs[end_index]
                best_end_index = end_index

        span_probabilities.append(
            (
                best_prob,
                start_index,
                end_index,
            )
        )
    span_probabilities = sorted(span_probabilities, key=lambda t: -t[0])
    return span_probabilities[0:5]


def mask_passage(input_ids, tokenizer, masking_percentage):
    mask_id = tokenizer.mask_token_id
    dot_id = tokenizer.encode(".")[1]

    sep_id = tokenizer.sep_token_id
    random_mask = torch.rand(*input_ids.shape) < masking_percentage
    input_ids[random_mask] = mask_id

    return input_ids
