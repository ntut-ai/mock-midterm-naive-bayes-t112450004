
import numpy as np
from scipy.stats import norm

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.class_prior = None
        self.mean = None
        self.variance = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.class_prior = np.zeros(n_classes)
        self.mean = np.zeros((n_classes, n_features))
        self.variance = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_prior[i] = X_c.shape[0] / X.shape[0]
            self.mean[i, :] = X_c.mean(axis=0)
            self.variance[i, :] = X_c.var(axis=0)

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))

        for i, c in enumerate(self.classes):
            prior = np.log(self.class_prior[i])
            likelihood = np.sum(np.log(norm.pdf(X, self.mean[i, :], np.sqrt(self.variance[i, :]))), axis=1)
            probabilities[:, i] = prior + likelihood

        # Normalize to get actual probabilities (optional, argmax is sufficient for prediction)
        # probabilities = np.exp(probabilities)
        # probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)

        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]
