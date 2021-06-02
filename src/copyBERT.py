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
            return_attention_mask=True,
            truncation=True)
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

    '''
    param_grid = {
        'gradient_accumulation_steps': [1, 4, 8],
        'eta': [0.00005, 0.0001, 0.001],
        'hidden_dim': [100, 200, 300]}
    '''

    param_grid = {'gradient_accumulation_steps': [8], 'eta': [0.0001], 'hidden_dim': [300]}
    bestmod = utils.fit_classifier_with_hyperparameter_search(
        X, y, basemod, cv=3, param_grid=param_grid)

    return bestmod

from transformers import DistilBertTokenizerFast
from datasets import load_dataset, load_metric
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

class ConvoteDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

def copyBERT():
    print("Starting copybert")
    train_df = utils.convote2sst('../convote_v1.1/data_stage_one/training_set/')
    dev_df = utils.convote2sst('../convote_v1.1/data_stage_one/development_set/')
    test_df = utils.convote2sst('../convote_v1.1/data_stage_one/test_set/')

    train_labels = list(train_df['label'])
    train_texts = list(train_df['sentence'])
    val_labels = list(dev_df['label'])
    val_texts = list(dev_df['sentence'])
    test_labels = list(test_df['label'])
    test_texts = list(test_df['sentence'])

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = ConvoteDataset(train_encodings, train_labels)
    val_dataset = ConvoteDataset(val_encodings, val_labels)
    test_dataset = ConvoteDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    preds, label_ids, metrics = trainer.predict(test_dataset)
    print("PREDS:", preds)
    print("Label_ids", label_ids)
    print("METRICS:", metrics)
