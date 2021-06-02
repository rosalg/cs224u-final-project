from Model import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

class SimpleNN(Model):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.clf = MLPClassifier(hidden_layer_sizes=(50, 10, 2), solver='adam', tol=1e-5)  # , max_iter=1

    def transform(self, df : pd.DataFrame, test=False):
        corpus = df["Text"]
        if test:
            tfidf = self.vectorizer.transform(corpus).toarray()
        else:
            tfidf = self.vectorizer.fit_transform(corpus).toarray()
        punc = np.reshape(list(df["Punctuation"]), (len(df), -1))
        X = np.concatenate((tfidf, punc), axis=1)
        return X

    def train(self, df : pd.DataFrame):
        X = self.transform(df)
        self.clf = self.clf.fit(X, df["Vote"])
        print("nn fitted")

    def predict_votes(self, df : pd.DataFrame):
        new_X = self.transform(df, test=True)
        predicted = self.clf.predict(new_X)
        return predicted
