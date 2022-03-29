#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd


def load_datafile(filename, split):
    with open(filename, "r") as f:
        data = [json.loads(l) for l in f]
    for s in data:
        s["split"] = split
    return data


def get_normal_split(split_df, dev_percentage):
    no_dev_samples = round((dev_percentage / 100) * split_df.shape[0])
    dev_split_df = split_df.sample(n=no_dev_samples, random_state=42)
    train_split_df = split_df.drop(index=dev_split_df.index)
    return train_split_df, dev_split_df


def dump_to_file(split_df, split_type, filename):
    with open(f"data/{filename}_{split_type}.jsonl", "w") as f:
        for l in split_df.to_dict("records"):
            f.write(json.dumps(l, ensure_ascii=False))
            f.write("\n")


def main():
    train_data = load_datafile("data/train_ar.jsonl", split="train")
    eval_data = load_datafile("data/eval_ar.jsonl", split="eval")

    concat_data = train_data + eval_data

    df = pd.DataFrame(concat_data)
    df["answer"] = df["answers"].apply(
        lambda a_list: "|".join(
            [a["text"] for a in sorted(a_list, key=lambda l: l["start_char"])]
        )
    )
    df["passage_answer"] = df.apply(
        lambda row: f"{row['passage']}|{row['answer']}", axis=1
    )
    df["question_answer"] = df.apply(
        lambda row: f"{row['question']}|{row['answer']}", axis=1
    )

    questions_count_dict = {
        row["index"]: row["question"]
        for i, row in df["question"].value_counts().reset_index().iterrows()
    }

    print(len(eval_data) / (len(train_data) + len(eval_data)))

    # Having same (question or passage) & answer pairs repeated
    leakage_indomain_df = df[
        (df["passage_answer"].duplicated(keep=False))
        | (df["question_answer"].duplicated(keep=False))
    ].sort_values(by="passage_answer")
    leakage_indomain_df.shape[0], leakage_indomain_df.shape[0] / df.shape[0], len(
        leakage_indomain_df["passage"].unique()
    ), len(leakage_indomain_df["question"].unique())

    unique_passage_df = df.drop(index=leakage_indomain_df.index)
    unique_passage_df = unique_passage_df[
        ~unique_passage_df["passage"].duplicated(keep=False)
    ]
    print(unique_passage_df.shape)

    # TODO: Find questions that aren't part of the unique passage_df
    threshold = 3
    total_ood_df = unique_passage_df[
        unique_passage_df["question"].apply(lambda q: questions_count_dict[q])
        <= threshold
    ]
    print(total_ood_df.shape)

    context_ood_df = unique_passage_df[
        unique_passage_df["question"].apply(lambda q: questions_count_dict[q])
        > threshold
    ]
    print(context_ood_df.shape)

    non_leakage_indomain_df = df.drop(
        index=leakage_indomain_df.index.tolist() + unique_passage_df.index.tolist()
    )

    assert (
        leakage_indomain_df.shape[0]
        + non_leakage_indomain_df.shape[0]
        + total_ood_df.shape[0]
        + context_ood_df.shape[0]
        == df.shape[0]
    )

    # # Types of datasets
    # ### TODO: Fix the strategy
    # - `leakage_indomain_df`: Multiple questions/passages having the same answer for paraphrased questions or different similar passages
    #     - Split them into overlapping sets 13%/87% split
    # - `total_ood_df`: Hard rare questions
    #     - Split them into (non-overlapping questions) sets as 13%/87% split
    # - `context_ood_df`: Repeated questions on new passages
    #     - Split them into overlapping sets as 13%/87% split
    # - `non_leakage_indomain_df`: Repeated passages with different questions and answers
    #     - Split them

    dev_percentage = 13.3
    train_splits, dev_splits = [], []
    filenames = [
        "leakage_indomain",
        "context_ood",
        "non_leakage_indomain",
        "total_ood",
    ]
    dfs = [
        leakage_indomain_df,
        context_ood_df,
        non_leakage_indomain_df,
        total_ood_df,
    ]
    train_split_dfs = []
    eval_split_dfs = []
    for typed_df, filename in zip(dfs[:-1], filenames[:-1]):
        t_split, d_split = get_normal_split(
            typed_df.drop(["passage_answer", "question_answer"], axis=1), dev_percentage
        )
        dump_to_file(d_split, "eval", filename)
        train_split_dfs.append(t_split)
        eval_split_dfs.append(d_split)

    dump_to_file(pd.concat(train_split_dfs), "train", "faithful")
    dump_to_file(pd.concat(eval_split_dfs), "eval", "faithful")
    dump_to_file(
        dfs[-1].drop(["passage_answer", "question_answer"], axis=1),
        "eval",
        filenames[-1],
    )


if __name__ == "__main__":
    main()
