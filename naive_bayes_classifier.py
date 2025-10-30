
import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        self.prior_probs = {}
        self.conditional_probs = {}
        self.classes = None
        self.features = None

    def calculate_prior_probs(self, df):
        total_samples = len(df)
        self.classes = df['species'].unique()
        for cls in self.classes:
            class_samples = len(df[df['species'] == cls])
            self.prior_probs[cls] = class_samples / total_samples

    def calculate_conditional_probs(self, df):
        self.features = df.columns.tolist()
        self.features.remove('species')

        for cls in self.classes:
            self.conditional_probs[cls] = {}
            class_df = df[df['species'] == cls]
            for feature in self.features:
                mean = class_df[feature].mean()
                std = class_df[feature].std()
                self.conditional_probs[cls][feature] = {'mean': mean, 'std': std}

    def fit(self, df):
        self.calculate_prior_probs(df)
        self.calculate_conditional_probs(df)

    def predict_proba(self, sample):
        posteriors = {}
        for cls in self.classes:
            posterior = self.prior_probs[cls]
            for feature in self.features:
                mean = self.conditional_probs[cls][feature]['mean']
                std = self.conditional_probs[cls][feature]['std']

                # Handle zero standard deviation
                if std == 0:
                    std = 1e-9  # Add a small smoothing value

                exponent = -((sample[feature] - mean) ** 2) / (2 * (std ** 2))
                # Add a small value to probability_density to avoid log(0) if using log probabilities
                probability_density = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(exponent)
                posterior *= probability_density # Consider using log probabilities to avoid underflow

            posteriors[cls] = posterior
        return posteriors

    def predict(self, df_test):
        predictions = []
        for index, row in df_test.iterrows():
            posteriors = self.predict_proba(row)
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        return predictions
