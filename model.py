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
from utils import INTERROGATIVE_ARTICLES


class MultiTaskQAModel(nn.Module):
    def __init__(
        self,
        model_name,
        dropout_p=0.5,
        use_TAPT=False,
        embed_ner=False,
        embed_question=False,
    ):
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
        if embed_ner:
            self.ner_embeddings = nn.Embedding(
                num_embeddings=NER_N_LABELS + 1,
                embedding_dim=NER_EMBEDDING_SIZE,
                padding_idx=NER_EMBEDDING_PAD,
                max_norm=NER_EMBEDDING_MAX_NORM,
            )
        else:
            self.ner_embeddings = None
        if embed_question:
            self.question_embeddings = nn.Embedding(
                num_embeddings=len(INTERROGATIVE_ARTICLES) + 1,
                embedding_dim=QUESTION_EMBEDDING_SIZE,
                padding_idx=len(INTERROGATIVE_ARTICLES),
                max_norm=QUESTION_EMBEDDING_MAX_NORM,
            )
        else:
            self.question_embeddings = None
        self.ner_layer = nn.Linear(
            in_features=768, out_features=NER_N_LABELS, bias=True
        )
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
        ner_labels=None,
        question_ids=None,
    ):
        if self.ner_embeddings:
            # Fix the padding value for the embedding layer!
            ner_labels[ner_labels == CROSS_ENTROPY_IGNORE_INDEX] = NER_EMBEDDING_PAD
            # Compute ner embeddings
            ner_embeds = self.ner_embeddings(ner_labels)
            # Compute input embeddings from bert
            inputs_embeds = self.bert.embeddings(input_ids)
            # Add them
            assert inputs_embeds.shape == ner_embeds.shape
            inputs_embeds = inputs_embeds + ner_embeds

        if self.question_embeddings:
            # Compute question embeddings
            question_embeds = self.question_embeddings(question_ids).reshape(
                -1, 1, QUESTION_EMBEDDING_SIZE
            )
            # Compute input embeddings from bert
            if not self.ner_embeddings:
                inputs_embeds = self.bert.embeddings(input_ids)
            # Add question embedding to [CLS] embedding
            inputs_embeds[:, 0:1, :] += question_embeds

        if self.ner_embeddings or self.question_embeddings:
            input_ids = None

        bert_output = self.bert(
            input_ids=input_ids,
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
