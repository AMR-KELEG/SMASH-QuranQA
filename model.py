import json
import torch
from torch import nn
from transformers import pipeline
from data_utils import create_squad_examples, create_inputs_targets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from settings import *
from transformers import AutoModel


class MultiTaskQAModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout_p=0.5, use_TAPT=False):
        super().__init__()
        if use_TAPT:
            self.bert = AutoModel.from_pretrained(f"{model_name}")
        else:
            ner = pipeline("ner", model=model_name)
            self.bert = ner.model.bert
        if dropout_p != "0":
            self.dropout = nn.Dropout(p=dropout_p, inplace=False)
        else:
            self.dropout = None
        self.ner_layer = nn.Linear(in_features=768, out_features=2, bias=True)
        self.qa_layer = nn.Linear(in_features=768, out_features=2, bias=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )["last_hidden_state"]

        if self.dropout != None:
            bert_output = self.dropout(bert_output)

        # TODO: Do I need a dropout layer after BERT?
        qa_output = self.qa_layer(bert_output)
        ner_output = self.ner_layer(bert_output)

        return bert_output, qa_output, ner_output
