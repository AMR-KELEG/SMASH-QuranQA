import json
import os
import re
import sys

# TODO: Fix this!
sys.path.append("../")
sys.path.append("../quranqa/code/")

import numpy as np
from colorama import Fore
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler
from settings import GPU_ID, EPOCHS
from data_utils import (
    create_squad_examples,
    create_inputs_targets,
    load_dataset_as_tensors,
    load_samples_as_tensors,
)
from quranqa.code.quranqa22_eval import (
    normalize_text,
    remove_prefixes,
    pRR_max_over_ground_truths,
)
import argparse
from utils import find_interrogatives, get_spans, softmax
from itertools import groupby
import logging
from model import MultiTaskQAModel

logger = logging.getLogger("Inspect")
parser = argparse.ArgumentParser(description="Inspect the models.")
parser.add_argument(
    "--seed",
    default=0,
    help="The value of the random seed to use.",
)
parser.add_argument(
    "--epoch",
    default=EPOCHS,
    help="The value of the epoch at which the checkpoint was generated.",
)
parser.add_argument(
    "--question_first",
    default=False,
    action="store_true",
    help="Use question as segment A, and passage as segment B",
)
parser.add_argument(
    "--use_TAPT",
    default=False,
    action="store_true",
    help="Use the model further PT on quran.",
)
parser.add_argument("--desc", required=True, help="The description of the model.")
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
    "--model_name",
    default="CAMeL-Lab/bert-base-arabic-camelbert-ca",
    help="The name of the BERT model to fine-tune.",
)
args = parser.parse_args()

MODEL_NAME = args.model_name
# Load the tokenizer
tokenizer = BertWordPieceTokenizer(f"{MODEL_NAME}_/vocab.txt", lowercase=True)

# Load the trained model
model = MultiTaskQAModel(
    MODEL_NAME,
    use_TAPT=args.use_TAPT,
    embed_ner=args.embed_ner,
    embed_question=args.embed_question,
).to(device=GPU_ID)
model.load_state_dict(
    torch.load(f"checkpoints/weights_{args.desc}_seed_{args.seed}_{args.epoch}.pth")
)
model.eval()

question = "من هم الأنبياء المذكورين؟"
# question = "هل أرسل الله رسل إلي فرعون؟"
# question = "كيف عاقب الله فرعون؟"
# question = "كيف يرد عباد الرحمن علي الكافرين؟"
# question = "كيف اهلك الله قوم عاد؟"
# passage = "ووصى بها إبراهيم بنيه ويعقوب يا بني إن الله اصطفى لكم الدين فلا تموتن إلا وأنتم مسلمون"
# passage = "إنا أرسلنا إليكم رسولا شاهدا عليكم كما أرسلنا إلى فرعون رسولا. فعصى فرعون الرسول فأخذناه أخذا وبيلا. فكيف تتقون إن كفرتم يوما يجعل الولدان شيبا. السماء منفطر به كان وعده مفعولا. إن هذه تذكرة فمن شاء اتخذ إلى ربه سبيلا."
# passage = "وعباد الرحمن الذين يمشون على الأرض هونا وإذا خاطبهم الجاهلون قالوا سلاما. والذين يبيتون لربهم سجدا وقياما. والذين يقولون ربنا اصرف عنا عذاب جهنم إن عذابها كان غراما. إنها ساءت مستقرا ومقاما. والذين إذا أنفقوا لم يسرفوا ولم يقتروا وكان بين ذلك قواما. والذين لا يدعون مع الله إلها آخر ولا يقتلون النفس التي حرم الله إلا بالحق ولا يزنون ومن يفعل ذلك يلق أثاما. يضاعف له العذاب يوم القيامة ويخلد فيه مهانا. إلا من تاب وآمن وعمل عملا صالحا فأولئك يبدل الله سيئاتهم حسنات وكان الله غفورا رحيما. ومن تاب وعمل صالحا فإنه يتوب إلى الله متابا. والذين لا يشهدون الزور وإذا مروا باللغو مروا كراما. والذين إذا ذكروا بآيات ربهم لم يخروا عليها صما وعميانا. والذين يقولون ربنا هب لنا من أزواجنا وذرياتنا قرة أعين واجعلنا للمتقين إماما. أولئك يجزون الغرفة بما صبروا ويلقون فيها تحية وسلاما. خالدين فيها حسنت مستقرا ومقاما. قل ما يعبأ بكم ربي لولا دعاؤكم فقد كذبتم فسوف يكون لزاما."
passage = "وما محمد إلا رسول قد خلت من قبله الرسل أفإن مات أو قتل انقلبتم على أعقابكم ومن ينقلب على عقبيه فلن يضر الله شيئا وسيجزي الله الشاكرين"
# passage = " الحاقة. ما الحاقة. وما أدراك ما الحاقة. كذبت ثمود وعاد بالقارعة. فأما ثمود فأهلكوا بالطاغية. وأما عاد فأهلكوا بريح صرصر عاتية. سخرها عليهم سبع ليال وثمانية أيام حسوما فترى القوم فيها صرعى كأنهم أعجاز نخل خاوية. فهل ترى لهم من باقية."
# passage = "واتل عليهم نبأ إبراهيم. إذ قال لأبيه وقومه ما تعبدون. قالوا نعبد أصناما فنظل لها عاكفين. قال هل يسمعونكم إذ تدعون. أو ينفعونكم أو يضرون. قالوا بل وجدنا آباءنا كذلك يفعلون. قال أفرأيتم ما كنتم تعبدون. أنتم وآباؤكم الأقدمون. فإنهم عدو لي إلا رب العالمين. الذي خلقني فهو يهدين. والذي هو يطعمني ويسقين. وإذا مرضت فهو يشفين. والذي يميتني ثم يحيين. والذي أطمع أن يغفر لي خطيئتي يوم الدين. رب هب لي حكما وألحقني بالصالحين. واجعل لي لسان صدق في الآخرين. واجعلني من ورثة جنة النعيم. واغفر لأبي إنه كان من الضالين. ولا تخزني يوم يبعثون. يوم لا ينفع مال ولا بنون. إلا من أتى الله بقلب سليم."
data = [
    {
        "question": question,
        "passage": passage,
        "pq_id": "PLACEHOLDER",
        "answers": [],
    }
]
(
    input_word_ids,
    input_mask,
    input_type_ids,
    question_ids,
    ner_labels,
) = load_samples_as_tensors(
    data, "Loading sample", tokenizer, question_first=args.question_first
)
outputs = model(
    input_ids=input_word_ids.to(GPU_ID),
    attention_mask=input_mask.to(GPU_ID),
    token_type_ids=input_type_ids.to(GPU_ID),
    question_ids=question_ids.to(GPU_ID),
    ner_labels=ner_labels.to(GPU_ID),
)

start_logits, end_logits = outputs[1].split(1, dim=-1)

start_logits = start_logits.squeeze(-1)
end_logits = end_logits.squeeze(-1)
pred_start, pred_end = (
    start_logits.detach().cpu().numpy(),
    end_logits.detach().cpu().numpy(),
)
# ner_pred = outputs[2].detach().cpu().numpy().reshape(-1, 2).argmax(axis=1)

test_sample = create_squad_examples(
    data, "Sample", tokenizer, question_first=args.question_first
)[0]
try:
    offsets = test_sample.context_token_to_char
except:
    # TODO: This is a hack added by Amr!
    # Investigate the reason for this!
    offsets = []

start_prob = softmax(pred_start, temperature=3)
end_prob = softmax(pred_end, temperature=3)
# for st_p, en_p, offset, is_ner in zip(start_prob, end_prob, offsets, ner_pred):
#     print(test_sample.context[offset[0] : offset[1]], st_p, en_p, is_ner)
for st_p, en_p, offset in zip(
    start_prob,
    end_prob,
    offsets,
):
    print(
        test_sample.context[offset[0] : offset[1]],
        st_p,
        en_p,
    )

print()
start = np.argmax(pred_start)
end = np.argmax(pred_end)
if end == len(offsets) - 1:
    end -= 1
print(offsets)
pred_char_start = offsets[start][0]
if end < len(offsets) and end >= start:
    print("Character range:", pred_char_start, offsets[end][1])
    pred_ans = test_sample.context[pred_char_start : offsets[end][1]]
else:
    pred_ans = test_sample.context[pred_char_start:]
    print("Fallback!")

print(start, end)
print(pred_ans)
