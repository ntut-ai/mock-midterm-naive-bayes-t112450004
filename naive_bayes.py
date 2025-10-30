
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to convert a column of string values to integer codes
def str_column_to_int(dataset, column):
    class_values = dataset[column].unique()
    lookup = dict()
    for i, value in enumerate(class_values):
        lookup[value] = i
    dataset[column] = dataset[column].map(lookup)
    return lookup

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.class_priors_ = defaultdict(float)
        self.mean_ = defaultdict(lambda: np.zeros(X.shape[1]))
        self.var_ = defaultdict(lambda: np.zeros(X.shape[1]))

        n_samples, n_features = X.shape
        for c in self.classes_:
            X_c = X[y == c]
            self.class_priors_[c] = len(X_c) / n_samples
            self.mean_[c] = X_c.mean(axis=0)
            self.var_[c] = X_c.var(axis=0)

    def _pdf(self, X, mean, var):
        epsilon = 1e-9  # Small value to prevent division by zero in variance
        numerator = np.exp(-((X - mean) ** 2) / (2 * (var + epsilon)))
        denominator = np.sqrt(2 * np.pi * (var + epsilon))
        return np.log(numerator / denominator)

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]): # Iterate through each data point using index
            x = X.iloc[i].values # Access data point using iloc
            posteriors = []
            for c in self.classes_:
                prior = np.log(self.class_priors_[c])
                likelihood = np.sum(self._pdf(x, self.mean_[c], self.var_[c]))
                posterior = prior + likelihood
                posteriors.append(posterior)
            y_pred.append(self.classes_[np.argmax(posteriors)])
        return np.array(y_pred)

# Load the datasets
iris_df = pd.read_csv("/content/mock-midterm-naive-bayes-t112450004/data/IRIS.csv")
iris_test_df = pd.read_csv("/content/mock-midterm-naive-bayes-t112450004/data/iris_test.csv")

# Prepare the data
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['species']
X_test = iris_test_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = iris_test_df['species'] # Keep y_test for evaluation

# Train the model
model = GaussianNaiveBayes()
model.fit(X, y) # Train on the full training data

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model (Optional - can be done in the notebook if preferred)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型在測試集上的準確度為: {accuracy:.4f}")

# Export predictions
predictions_df = X_test.copy()
predictions_df['predicted_species'] = y_pred
predictions_df.to_csv("/content/mock-midterm-naive-bayes-t112450004/predictions.csv", index=False)

print("預測結果已匯出至 /content/mock-midterm-naive-bayes-t112450004/predictions.csv")
