import pandas as pd
import numpy as np
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
import torch
import utils
import sst


class HfBertClassifierModel(nn.Module):
    def __init__(self, n_classes, weights_name='bert-base-cased'):
        super().__init__()
        self.n_classes = n_classes
        self.weights_name = weights_name
        self.bert = BertModel.from_pretrained(self.weights_name)
        self.bert.train()
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        # The only new parameters -- the classifier:
        self.classifier_layer = nn.Linear(
            self.hidden_dim, self.n_classes)

    def forward(self, indices, mask):
        reps = self.bert(
            indices, attention_mask=mask)
        return self.classifier_layer(reps.pooler_output)


class HfBertClassifier(TorchShallowNeuralClassifier):
    def __init__(self, weights_name, *args, **kwargs):
        self.weights_name = weights_name
        self.tokenizer = BertTokenizer.from_pretrained(self.weights_name)
        super().__init__(*args, **kwargs)
        self.params += ['weights_name']

    def build_graph(self):
        return HfBertClassifierModel(self.n_classes_, self.weights_name)

    def build_dataset(self, X, y=None):
        data = self.tokenizer.batch_encode_plus(
            X,
            max_length=None,
            add_special_tokens=True,
            padding='longest',
            return_attention_mask=True)
        indices = torch.tensor(data['input_ids'])
        mask = torch.tensor(data['attention_mask'])
        if y is None:
            dataset = torch.utils.data.TensorDataset(indices, mask)
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(indices, mask, y)
        return dataset


def bert_fine_tune_phi(text):
    return text


def fit_hf_bert_classifier_with_hyperparameter_search(X, y):
    basemod = HfBertClassifier(
        weights_name='bert-base-cased',
        batch_size=8,  # Small batches to avoid memory overload.
        max_iter=1,  # We'll search based on 1 iteration for efficiency.
        n_iter_no_change=5,   # Early-stopping params are for the
        early_stopping=True)  # final evaluation.

    param_grid = {
        'gradient_accumulation_steps': [1, 4, 8],
        'eta': [0.00005, 0.0001, 0.001],
        'hidden_dim': [100, 200, 300]}

    bestmod = utils.fit_classifier_with_hyperparameter_search(
        X, y, basemod, cv=3, param_grid=param_grid)

    return bestmod

def gilBERT():
    train_df = utils.convote2sst('/convote_v1.1/data_stage_one/training_set')
    print(train_df.head())
    dev_df = utils.convote2sst('/convote_v1.1/data_stage_one/development_set')
    test_df = utils.convote2sst('/convote_v1.1/data_stage_one/test_set')
    bert_classifier_xval = sst.experiment(
        train_df,
        bert_fine_tune_phi,
        fit_hf_bert_classifier_with_hyperparameter_search,
        assess_dataframes=dev_df,
        vectorize=False)  # Pass in the BERT hidden state directly!
    optimized_bert_classifier = bert_classifier_xval['model']

    def fit_optimized_hf_bert_classifier(X, y):
        optimized_bert_classifier.max_iter = 1000
        optimized_bert_classifier.fit(X, y)
        return optimized_bert_classifier

    _ = sst.experiment(
        train_df,
        bert_fine_tune_phi,
        fit_optimized_hf_bert_classifier,
        assess_dataframes=test_df,
        vectorize=False,
        verbose=True)
    print("Score: ")
    print(_['score'])
