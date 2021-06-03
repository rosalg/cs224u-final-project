import pandas as pd
import numpy as np
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from datasets import load_metric
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
import torch
import utils
import sst

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

def gilBERT():
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
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        gradient_accumulation_steps=1
    )
    metric = load_metric("f1")

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
