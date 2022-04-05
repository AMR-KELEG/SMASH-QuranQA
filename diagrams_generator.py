import json
import pandas as pd
from utils import find_interrogatives
from data_utils import download_dataset

import seaborn as sns
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display


def load_dataset(filename):
    with open(filename, "r") as f:
        return pd.DataFrame([json.loads(l) for l in f])


def generate_question_type_diagram(train_df, eval_df, normalize=False):
    for df in [train_df, eval_df]:
        df["question_type"] = df["question"].apply(lambda q: find_interrogatives(q))
        assert df["question_type"].value_counts().sum() == df.shape[0]
    train_type_counts = (
        train_df["question_type"].value_counts(normalize=normalize).reset_index()
    )
    eval_type_counts = (
        eval_df["question_type"].value_counts(normalize=normalize).reset_index()
    )

    merged_df = pd.merge(
        left=train_type_counts, right=eval_type_counts, how="outer", on="index"
    )
    merged_df.columns = ["Interrogative article", "Count (train)", "Count (dev)"]
    merged_df.fillna(0, inplace=True)
    merged_df["Interrogative article"] = merged_df["Interrogative article"].apply(
        lambda l: get_display(arabic_reshaper.reshape(l))
    )

    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=False, sharey=True, figsize=(6.3, 4)
    )
    g1 = sns.barplot(
        x="Interrogative article",
        y="Count (train)",
        data=merged_df,
        log=True,
        color="#CBC3E3",
        ax=axes[0],
    )

    g2 = sns.barplot(
        x="Interrogative article",
        y="Count (dev)",
        data=merged_df,
        color="#CBC3E3",
        ax=axes[1],
    )

    return fig


if __name__ == "__main__":
    # Download the datasets
    download_dataset()

    train_df = load_dataset("data/train_ar.jsonl")
    eval_df = load_dataset("data/eval_ar.jsonl")

    fig = generate_question_type_diagram(train_df, eval_df, normalize=False)
    fig.savefig("output_plots/question_types.pdf", bbox_inches="tight")
