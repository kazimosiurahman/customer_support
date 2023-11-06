import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd


class MeanEmbeddingVectorizer(object):

    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def cross_validate_models(models, X_train, y_train, cv=5):
    cv_df = pd.DataFrame(index=range(cv * len(models)))
    entries = []
    for model in models:
        try:
          model_name = model.__class__.__name__
          accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv)
          for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
        except Exception as e:
            entries.append((model_name, fold_idx, 0))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    return cv_df