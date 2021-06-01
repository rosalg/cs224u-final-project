from Model import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

class SimpleNN(Model):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.clf = MLPClassifier(hidden_layer_sizes=(50, 10, 2), solver='adam', tol=1e-5)

    def train(self, df : pd.DataFrame):
        corpus = df["Text"]
        tfidf = TfidfVectorizer().fit_transform(corpus).toarray()
        punc = np.reshape(list(df["Punctuation"]), (len(df), -1))
        X = np.concatenate((tfidf, punc), axis=1)
        self.clf = self.clf.fit(X, df["Vote"])
        print("nn fitted")

    def predict_votes(self, df : pd.DataFrame):
        new_corpus = df["Text"]
        new_X = self.vectorizer.transform(new_corpus)
        predicted = self.clf.predict(new_X)
        return predicted
