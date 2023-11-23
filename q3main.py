import pandas as pd
import numpy as np
import os

# Specifying a relative path to the local directory
dirname = os.path.dirname(__file__)
filepath = os.path.join(dirname, 'dataset')

# Read the csv files
x_train = pd.read_csv(filepath + '\\X_train.csv', delimiter=' ').values
y_train = pd.read_csv(filepath + '\\y_train.csv', header=None).values.ravel()
x_test = pd.read_csv(filepath + '\\X_test.csv', delimiter=' ').values
y_test = pd.read_csv(filepath + '\\y_test.csv', header=None).values.ravel()


# Function to calculate probabilities
def calculate_probabilities_multinomial_nosmoothing(x_features_train, y_labels_train):
    """Calculates the prior probability of all labels and
    likelihood probability for all features given all labels.
    In summary it calculates all instances of prior probability 
    P(Y=yk) and likelihood prob P(Xj = xj | Y = yk)
    Args:
        x_features_train (numpy.2darray): number of features in each instance of the training data 
        y_labels_train (numpy.1darray): labels of each instance of the training data

    Returns:
        prior_prob (numpy.1darray): prior probability for all labels
        likelihood_prob (numpy.2darray): all instances of likelihood probabilites
    """
    num_classes = len(np.unique(y_labels_train))
    num_features = x_features_train.shape[1]

    # Initialize parameters
    class_counts = np.zeros(num_classes)
    feature_counts = np.zeros((num_classes, num_features))

    # Calculate class and feature counts
    for index, class_instance in enumerate(y_labels_train):
        class_counts[class_instance] += 1
        feature_counts[class_instance] += x_features_train[index]

    prior_prob = (class_counts) / (len(y_labels_train))
    likelihood_prob = (feature_counts) / (np.sum(feature_counts, axis=1)[:, np.newaxis])

    return prior_prob, likelihood_prob

# Function to predict
def predict_multinomial_nosmoothing(x_features_test, prior_probabilites, likelihood_probabilities):
    log_likelihood_prob = np.where(likelihood_probabilities < 0.0000000001, -10**12, np.log(likelihood_probabilities ,where = likelihood_probabilities > 0.0000000001))
    log_probs = np.log(prior_probabilites) + np.dot(x_features_test, log_likelihood_prob.T)
    return np.argmax(log_probs, axis=1)

# Train the Multinomial Naive Bayes model
prior_prob, likelihood_prob = calculate_probabilities_multinomial_nosmoothing(x_train, y_train)
y_pred = predict_multinomial_nosmoothing(x_test, prior_prob, likelihood_prob)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.3f}')

# Confusion Matrix
conf_matrix = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))))
for i in range(len(y_test)):
  conf_matrix[y_pred[i], y_test[i]] += 1

print('Confusion Matrix:')
print(conf_matrix)

# Function to calculate probabilities
def calculate_probabilities_multinomial_smoothing(x, y, alpha = 1):
    num_classes = len(np.unique(y))
    num_features = x.shape[1]

    # Initialize parameters
    class_counts = np.zeros(num_classes)
    feature_counts = np.zeros((num_classes, num_features))

    # Calculate class and feature counts
    for i in range(len(y)):
        class_counts[y[i]] += 1
        feature_counts[y[i]] += x[i]

    prior_prob = (class_counts) / (len(y))
    likelihood_prob = (feature_counts + alpha) / (np.sum(feature_counts, axis=1)[:, np.newaxis] + alpha * num_features)

    return prior_prob, likelihood_prob

# Function to predict
def predict_multinomial_smoothing(x, prior_prob, likelihood_prob):
    log_likelihood_prob = np.where(likelihood_prob < 0.0000000001, -10**12, np.log(likelihood_prob,where = likelihood_prob > 0.0000000001))
    log_probs = np.log(prior_prob) + np.dot(x, log_likelihood_prob.T)
    return np.argmax(log_probs, axis=1)

# Train the Multinomial Naive Bayes model
prior_prob, likelihood_prob = calculate_probabilities_multinomial_smoothing(x_train, y_train)
y_pred = predict_multinomial_smoothing(x_test, prior_prob, likelihood_prob)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.3f}')

# Confusion Matrix
conf_matrix = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))))
for i in range(len(y_test)):
    conf_matrix[y_pred[i], y_test[i]] += 1

print('Confusion Matrix:')
print(conf_matrix)

# Function to calculate probabilities
def calculate_probabilities_bernoulli(x, y, alpha = 1):
    num_classes = len(np.unique(y))
    num_features = x.shape[1]

    # Initialize parameters
    class_counts = np.zeros(num_classes)
    feature_counts = np.zeros((num_classes, num_features))

    # Calculate class and feature counts
    for i in range(len(y)):
        class_counts[y[i]] += 1
        feature_counts[y[i]] += [1 if x[i, j] > 0 else 0 for j in range(num_features)]

    prior_prob = (class_counts) / (len(y))
    likelihood_prob = (feature_counts + alpha) / (class_counts.reshape(-1, 1) + alpha * 2)

    return prior_prob, likelihood_prob

# Function to predict
def predict_bernoulli(x, prior_prob, likelihood_prob):
    x_binary = (x > 0).astype(int)
    log_likelihood_prob = np.where(likelihood_prob < 0.0000000001, -10**12, np.log(likelihood_prob,where = likelihood_prob > 0.0000000001))
    log_probs = np.log(prior_prob) + np.dot(x_binary, log_likelihood_prob.T) + np.dot(1 - x_binary, np.log(1 - likelihood_prob).T)
    return np.argmax(log_probs, axis=1)

# Train the Multinomial Naive Bayes model
prior_prob, likelihood_prob = calculate_probabilities_bernoulli(x_train, y_train)
y_pred = predict_bernoulli(x_test, prior_prob, likelihood_prob)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.3f}')

# Confusion Matrix
conf_matrix = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))))
for i in range(len(y_test)):
    conf_matrix[y_pred[i], y_test[i]] += 1

print('Confusion Matrix:')
print(conf_matrix)
