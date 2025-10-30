
import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.feature_means = {}
        self.feature_variances = {}
        self.classes = None

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        n_samples, n_features = X_train.shape

        for cls in self.classes:
            X_cls = X_train[y_train == cls]
            self.class_priors[cls] = len(X_cls) / n_samples
            self.feature_means[cls] = X_cls.mean(axis=0)
            self.feature_variances[cls] = X_cls.var(axis=0)

    def _gaussian_probability(self, x, mean, variance):
        # Add a small epsilon to variance to avoid division by zero
        variance = variance + 1e-6
        exponent = -((x - mean) ** 2) / (2 * variance)
        numerator = np.exp(exponent)
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

    def predict_proba(self, X_test):
        n_samples = X_test.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))

        for i, cls in enumerate(self.classes):
            prior = self.class_priors[cls]
            likelihood = np.prod(self._gaussian_probability(X_test, self.feature_means[cls], self.feature_variances[cls]), axis=1)
            probabilities[:, i] = prior * likelihood

        # Normalize probabilities
        row_sums = probabilities.sum(axis=1)[:, np.newaxis]
        probabilities = probabilities / row_sums
        return probabilities

    def predict(self, X_test):
        probabilities = self.predict_proba(X_test)
        return self.classes[np.argmax(probabilities, axis=1)]
