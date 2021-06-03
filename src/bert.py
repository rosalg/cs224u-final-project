import torch
import sst

from transformers import BertModel, BertTokenizer
import pandas as pd
import numpy
import codecs
from Model import *
from sklearn.feature_extraction.text import TfidfVectorizer

from torch_rnn_classifier import TorchRNNModel
from torch_rnn_classifier import TorchRNNClassifier
from torch_rnn_classifier import TorchRNNClassifierModel
import json
import utils


class Bert(Model):
    def __init__(self):
        self.COUNT = 0
        self.vectorizer = TfidfVectorizer()
        self.clf = None
        weights_name = 'bert-base-cased'
        self.bert_tokenizer = BertTokenizer.from_pretrained(weights_name)
        self.bert_model = BertModel.from_pretrained(weights_name)

    def bert_phi(self, text, test=False):
        input_ids = self.bert_tokenizer.encode(text, add_special_tokens=False, truncation = True)
        print(input_ids[:2])
        Ids = torch.tensor([input_ids])
        with torch.no_grad():
            reps = self.bert_model(Ids)
            return reps.last_hidden_state.squeeze(0).numpy()
        
    def transform_df(self, df : pd.DataFrame, test = False):
        new_df = pd.DataFrame()
        new_df["sentence"] = df["Text"]
        if not test:
            new_df["label"] = df["Vote"]
        print(new_df)
        return new_df

    def get_model(self, X, y):
        mod = TorchRNNClassifier(
            vocab=[],
            hidden_dim=300,
            early_stopping=True,
            use_embedding=False
        )
        mod.fit(X, y)
        return mod

    def train(self, df : pd.DataFrame, eval_df = None):
        train_df = utils.convote2sst('../convote_v1.1/data_stage_one/training_set/').head(20)
        # print(train_df.head())
        dev_df = utils.convote2sst('../convote_v1.1/data_stage_one/development_set/').head(20)
        train_dataset = self.transform_df(df.head(20))
        # train_dataset = sst.build_dataset(self.transform_df(df.head()), self.bert_phi, vectorize=False)
        # print(type(train_dataset))
        # json.dump(train_dataset.tolist(), codecs.open("bert_train.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
        # try: 
        #     with open("bert_train.json", "w") as outfile: 
        #         json.dump(train_dataset, outfile)
        # except: pass
        # print("dataset built")

        # eval_dataset = sst.build_dataset(transform_df(eval_df), bert_phi, vectorize=False)
        val = sst.experiment(
            train_df, #train set
            self.bert_phi, #bert_phi
            self.get_model, #get_model
            # assess_dataframes=[eval_dataset], #eval set
            assess_dataframes = dev_df,
            vectorize=False,
        )
        self.clf = val
        return val

    def predict_one_custom(self, text):
        rnn_experiment = self.clf
        feats = [rnn_experiment['phi'](text)]
        preds = rnn_experiment['model'].predict(feats)
        return preds[0]

    def predict_votes(self, df : pd.DataFrame):
        test_df = utils.convote2sst('../convote_v1.1/data_stage_one/test_set/').head(20)
        # new_df = self.transform_df(df.head(20), test=True)
        test_df['prediction'] = test_df['sentence'].apply(self.predict_one_custom)
        # print(new_df['prediction'])
        return list(test_df.prediction.values)
        # preds = self.clf['model'].predict(self.transform_df(df, test=True))
        # return preds