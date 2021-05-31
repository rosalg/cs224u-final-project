from Model import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

class SimpleSVM(Model):
    def __init__(self):
        self.vectorizer = None
        self.clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))

    def train(self, df : pd.DataFrame):
        corpus = df["Text"]
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(corpus)
        self.clf = self.clf.fit(X, df["Vote"])

    def predict_votes(self, df : pd.DataFrame):
        new_corpus = df["Text"]
        new_X = self.vectorizer.transform(new_corpus)
        predicted = self.clf.predict(new_X)
        return predicted
