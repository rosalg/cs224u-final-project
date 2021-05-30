import pandas as pd
import numpy as np

class Model:
    def __init__(self):
        pass

    def train(self, df : pd.DataFrame):
        raise NotImplementedError()

    def predict_votes(self, df : pd.DataFrame):
        raise NotImplementedError()
