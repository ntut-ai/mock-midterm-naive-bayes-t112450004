
import numpy as np
import math

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        # print('[%s] => %d' % (value, i)) # Avoid printing during test execution

    for row in dataset:
        row[column] = lookup[row[column]]

    return lookup


def separate_by_class(dataset):
    """Separates the dataset by class.

    Args:
        dataset: A list of lists where the last element of each
                 inner list is the class label.

    Returns:
        A dictionary where keys are class labels and values are lists of instances
        belonging to that class.
    """
    separated = {}
    for row in dataset:
        class_value = row[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(row)
    return separated

def mean(numbers):
    """Calculates the mean of a list of numbers."""
    if not numbers:
        return 0.0
    # Ensure all elements are numeric, handling potential string numbers
    numeric_numbers = [float(x) for x in numbers if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()]
    if not numeric_numbers:
        return 0.0 # Return 0 if no numeric data found after filtering
    return sum(numeric_numbers) / float(len(numeric_numbers))

def stdev(numbers):
    """Calculates the standard deviation of a list of numbers."""
    numeric_numbers = [float(x) for x in numbers if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()]
    if len(numeric_numbers) <= 1:
        return 0.0
    avg = mean(numeric_numbers)
    variance = sum([pow(x - avg, 2) for x in numeric_numbers]) / float(len(numeric_numbers) - 1)
    return math.sqrt(variance)

def summarize(dataset):
    """Calculates summary statistics (mean and stdev) for each attribute in a dataset.
    This function expects a dataset where each inner list represents a feature column.
    """
    # Filter out non-numeric columns before calculating summaries
    numeric_dataset = []
    for column in dataset:
        # Check if a column contains at least one numeric value before processing
        if any(isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit() for x in column):
            numeric_dataset.append(column)

    summaries = [(mean(attribute), stdev(attribute)) for attribute in numeric_dataset]
    return summaries


def summarize_by_class(dataset):
    """Calculates summary statistics for each attribute, separated by class.
    The input dataset is expected to have instances as rows and features + class as columns.
    """
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        # Extract only feature columns (excluding the last column which is the class label)
        # and ensure they are lists of numerical values
        instance_features = [instance[:-1] for instance in instances]

        # Transpose the feature columns to get lists of values for each feature
        if not instance_features: # Handle empty instance_features list
            continue
        instance_features_transposed = list(zip(*instance_features))

        summaries[class_value] = summarize(instance_features_transposed)
    return summaries

def calculate_probability(x, mean, stdev):
    """Calculates the Gaussian probability density function for a given value."""
    if stdev == 0:
        return 1.0 if x == mean else 0.0 # Handle zero standard deviation
    # Ensure x is numeric before calculation
    try:
        x = float(x)
    except (ValueError, TypeError):
        return 1e-9 # Return a small probability for non-numeric input


    exponent = math.exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, row):
    """Calculates the probabilities of predicting each class for a given row.
       Expects summaries to contain (mean, stdev, count) for each feature.
       Accepts a single row as input.
    """
    probabilities = {}
    # Assumes summaries contain count as the third element in feature summaries
    # Need a way to get total instances, assuming summaries[label][0][2] holds class instance count
    total_rows = sum([summaries[label][0][2] for label in summaries])

    for class_value, class_summaries in summaries.items():
        # Prior probability
        probabilities[class_value] = float(summaries[class_value][0][2]) / float(total_rows)

        # Multiply by the probability of each feature value
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            # Ensure row has enough features
            if i < len(row):
                 probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
            else:
                # Handle missing features in row
                probabilities[class_value] *= 1e-9 # Assign small probability


    return probabilities


# Modified nb_train to match the original nb_test.py's call signature
def nb_train(train_data):
    """Trains the Naive Bayes model.
       Accepts a single dataset argument (features + labels) as expected by nb_test.py

    Args:
        train_data: Training dataset (list of lists, last element is label).

    Returns:
        A dictionary containing the trained model parameters (summaries).
    """
    # Need to include counts in summaries for calculate_class_probabilities
    # Let's adjust summarize_by_class to return summaries with counts

    # Re-implementing summarize and summarize_by_class to include counts
    def summarize_with_count(dataset_subset):
        """Calculates summary statistics (mean, stdev, and count) for each attribute in a dataset subset."""
        summaries = []
        if not dataset_subset:
            return summaries

        num_features = len(dataset_subset[0]) -1 # Exclude the class label

        for i in range(num_features):
            feature_values = [instance[i] for instance in dataset_subset]
            numeric_values = [float(x) for x in feature_values if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()]

            if len(numeric_values) > 0:
                col_mean = mean(numeric_values)
                col_stdev = stdev(numeric_values)
                col_count = len(numeric_values)
                summaries.append((col_mean, col_stdev, col_count))
            else:
                summaries.append((0.0, 0.0, 0))

        return summaries


    def summarize_by_class_with_count(dataset):
        """Calculates summary statistics (mean, stdev, count) for each attribute, separated by class."""
        separated = separate_by_class(dataset)
        summaries = {}
        for class_value, instances in separated.items():
            # Pass the instances for each class to summarize_with_count
            summaries[class_value] = summarize_with_count(instances)
        return summaries

    # Use the summarize_by_class function that returns summaries with counts
    summaries = summarize_by_class_with_count(train_data)
    return summaries


# Modified nb_predict to match the original nb_test.py's call signature
def nb_predict(model, row):
    """Makes a prediction for a single instance using the trained Naive Bayes model.
       Accepts model and a single row as expected by nb_test.py

    Args:
        model: The trained model parameters (dictionary of summaries with counts).
        row: A single instance (list of feature values).

    Returns:
        The predicted class label (integer ID).
    """
    # Calculate probabilities using the function that expects summaries with counts and a single row
    probabilities = calculate_class_probabilities(model, row)

    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label # Return the integer ID as expected by nb_test.py


# The original naive_bayes.py had a __main__ block that was not compatible
# with how nb_train and nb_predict were defined in the notebook.
# Keeping a simple pass here or adding example usage is fine, but the original
# argument parsing and execution logic is not needed if using this as a module.
if __name__ == '__main__':
    # Example usage (you can add your own test cases here)
    # This block will only run if the script is executed directly
    pass
