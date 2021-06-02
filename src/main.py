import os
import re
import pandas as pd
import numpy as np
import sklearn
import argparse
from sklearn.preprocessing import StandardScaler
from Model import *
from Baseline import *
from SimpleNN import *
from bert import *
from SimpleSVM import *
from gilBERT import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', metavar='N', type=str, help='Model to run tests on.', required=True)
parser.add_argument('--print_basic', type=bool, default=False, help="Print basic training data information.")

CONVOTE_DATA_DIR = "../convote_v1.1/data_stage_one/"

BASELINES = {"baseline": Baseline, "simple_nn": SimpleNN, "simple_svm": SimpleSVM, "gilbert": gilBERT, "bert" : Bert}

def parse_convote_data(base_path):
    file_names = os.listdir(base_path)
    data = []
    for fn in file_names:
        # Get metadata; https://www.cs.cornell.edu/home/llee/data/convote/README.v1.1.txt
        m = re.match(r"(?P<bill>\d\d\d)_(?P<speaker>\d\d\d\d\d\d)_"
                     + r"(?P<page_num>\d\d\d\d)(?P<speech_num>\d\d\d)_"
                     + r"(?P<party>\w)(?P<mentioned>\w)(?P<vote>\w)\.txt", fn)
        bill = int(m.group("bill"))
        speaker = int(m.group("speaker"))
        bill_directly_mentioned = m.group("mentioned")
        vote = m.group("vote")
        # Get text
        with open(base_path + fn) as f:
            text = f.read()
        punctuation = 0
        if ("?" in text):
            punctuation = 1
        if ("!" in text):
            punctuation = 2

        # Save to dict
        data.append([bill, speaker, bill_directly_mentioned, text, vote, punctuation])  # base features

    df = pd.DataFrame(np.array(data), columns=["Bill number", "Speaker", "Bill mentioned",
                                               "Text", "Vote", "Punctuation"])
    sc = StandardScaler()
    df[["Punctuation"]] = sc.fit_transform(df[["Punctuation"]])
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    base_path = CONVOTE_DATA_DIR + "training_set/"
    df = parse_convote_data(base_path)
    if args.print_basic:
        print(df.head())
        print(len(df))

        unique_bills = np.unique(df["Bill number"])
        print("Number of debates:", len(unique_bills))
        for ub in unique_bills[:3]:
            print("Bill #:", ub)
            speeches = df[df["Bill number"] == ub]
            print("# of speeches:", len(speeches))
            unique_speakers = np.unique(speeches["Speaker"])
            print("# of speakers:", len(unique_speakers))

    test_base_path = CONVOTE_DATA_DIR + "test_set/"
    testing_df = parse_convote_data(test_base_path)

    if args.model in BASELINES:
        print("Running model: ", args.model)
        model = BASELINES[args.model]()
        if args.model != "gilbert":
            model.train(df)
            predicted = model.predict_votes(testing_df.drop("Vote", axis=1))

            num_corr = 0
            num_tot = 0
            assert (len(predicted) == len(testing_df["Vote"]))
            for i in range(len(predicted)):
                num_tot += 1
                if predicted[i] == testing_df["Vote"][i]: num_corr += 1
            print("Accuracy of", args.model, ":",num_corr / num_tot)
    else:
        print("Please choose one of the valid baselines to run: ", BASELINES.keys())
