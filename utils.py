from scipy.special import softmax

INTERROGATIVE_ARTICLES = (
    "لماذا كيف بأي بماذا من كم كيف ما ماذا هل اين متى".split()
    + ["من هو", "من الذي",]
    + ["ما المخلوقات", "ما هي انواع", "ما هي اسماء"]
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

    return final_interrogatives[0] if final_interrogatives else ""


def get_spans(start_logits, end_logits):
    start_probs = softmax(start_logits)
    end_probs = softmax(end_logits)

    span_probabilities = []
    for start_index in range(0, len(start_probs)):
        for end_index in range(start_index, len(end_probs)):
            span_probabilities.append(
                (
                    start_probs[start_index] * end_probs[end_index],
                    start_index,
                    end_index,
                )
            )
    span_probabilities = sorted(span_probabilities)
    return span_probabilities[-5:]
