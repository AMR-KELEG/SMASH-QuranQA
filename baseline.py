import json
from data_utils import create_squad_examples, create_inputs_targets

with open("data/eval_ar.jsonl") as f:
    raw_eval_data = [json.loads(l) for l in f]

test_samples = create_squad_examples(
    raw_eval_data, "Creating test points", tokenizer=None
)
ids = [test_sample.question_id for test_sample in test_samples]
answers = [test_sample.context for test_sample in test_samples]

with open("data/smash_run00.json", "w") as f:
    submission = {
        id: [{"answer": answer, "rank": 1, "score": 1}]
        for id, answer in zip(ids, answers)
    }
    json.dump(submission, f, ensure_ascii=False)
