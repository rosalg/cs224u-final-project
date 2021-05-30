from Model import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class Baseline(Model):
    def __init__(self):
        self.vectorizer = None
        self.clf = None

    def train(self, df : pd.DataFrame):
        corpus = df["Text"]
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(corpus)
        self.clf = MultinomialNB().fit(X, df["Vote"])

    def predict_votes(self, df : pd.DataFrame):
        new_corpus = df["Text"]
        new_X = self.vectorizer.transform(new_corpus)
        predicted = self.clf.predict(new_X)
        return predicted
