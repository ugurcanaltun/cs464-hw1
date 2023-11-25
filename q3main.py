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


# Function to calculate multinomial prior and likelihood probabilities with no smoothing performed
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

    # Initialize the parameters
    class_counts = np.zeros(num_classes)
    feature_counts = np.zeros((num_classes, num_features))

    # Calculate class and feature counts
    for index, class_instance in enumerate(y_labels_train):
        class_counts[class_instance] += 1
        feature_counts[class_instance] += x_features_train[index]

    prior_prob_calculated = (class_counts) / (len(y_labels_train))
    likelihood_prob_calculated = (feature_counts) / (np.sum(feature_counts, axis=1)[:, np.newaxis])

    return prior_prob_calculated, likelihood_prob_calculated

# Function to predict via multinomial
def predict_multinomial_bayes(x_features_test, prior_probabilites, likelihood_probabilities):
    """In this function, we are making prediction according to multinomial naive bayes
    model. We are performing the following formula:
            yi = argmax( logP(Y=yk) + sum twj,i * logP(Xj/Y=yk), j=i to |V|)
                   y
    Args:
        x_features_test (numpy.2darray): the number of the features in the test set
        that we are going to use for prediction
        prior_probabilites (numpy.1darray): the prior probabilites that we previously calculated
        likelihood_probabilities (numpy.2darray): the likelihood probabilities that we 
        previously calculated 

    Returns:
        argmax(numpy.1darray): we create an array from the indices of the greatest values
        over axis 1, which indicates which class each instance in the test set according to
        our prediction
    """
    # Taking logs of the probabilities, taking probabilities that are near zero as -10**12 instead of
    # taking thier logs to avoid underflow
    log_likelihood_prob = np.where(likelihood_probabilities < 0.0000000001, -10**12, np.log(likelihood_probabilities ,where = likelihood_probabilities > 0.0000000001))
    #Doing matrix multiplication and making arrays suitable for matrix multiplication
    log_probs = np.log(prior_probabilites) + np.dot(x_features_test, log_likelihood_prob.T)
    return np.argmax(log_probs, axis=1)

# Train the Multinomial Naive Bayes with no smoothing
prior_prob, likelihood_prob = calculate_probabilities_multinomial_nosmoothing(x_train, y_train)
y_pred = predict_multinomial_bayes(x_test, prior_prob, likelihood_prob)

# Calculate the accuracy of multinomial bayes with no smoothing
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.3f}')

# Create the Confusion Matrix for multinomial bayes with no smoothing
conf_matrix = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))))
for index, label in enumerate(y_test):
    conf_matrix[y_pred[index], y_test[index]] += 1

print('Confusion Matrix:')
print(conf_matrix)

# Function to calculate multinomial prior and likelihood probabilities with smoothing performed
def calculate_probabilities_multinomial_smoothing(x_features_train, y_labels_train, alpha = 1):
    """Calculates the prior probability of all labels and
    likelihood probability for all features given all labels.
    In summary it calculates all instances of prior probability 
    P(Y=yk) and likelihood prob P(Xj = xj | Y = yk) by using a 
    fair Dirichlet prior
    Args:
        x_features_train (numpy.2darray): number of features in each instance of the training data 
        y_labels_train (numpy.1darray): labels of each instance of the training data
        alpha (int): Dirichlet prior constant. Defaults to 1.

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

    prior_prob_calculated = (class_counts) / (len(y_labels_train))
    likelihood_prob_calculated = (feature_counts + alpha) / (np.sum(feature_counts, axis=1)[:, np.newaxis] + alpha * num_features)

    return prior_prob_calculated, likelihood_prob_calculated

# Train the Multinomial Naive Bayes model with smoothing
prior_prob, likelihood_prob = calculate_probabilities_multinomial_smoothing(x_train, y_train)
y_pred = predict_multinomial_bayes(x_test, prior_prob, likelihood_prob)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.3f}')

# Create the Confusion Matrix for multinomial bayes with no smoothing
conf_matrix = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))))
for index, label in enumerate(y_test):
    conf_matrix[y_pred[index], y_test[index]] += 1

print('Confusion Matrix:')
print(conf_matrix)

# Function to calculate bernoulli prior and likelihood probabilities with smoothing performed
def calculate_probabilities_bernoulli(x_features_train, y_labels_train, alpha = 1):
    """Prior probabilities for each class label and likelihood probabilities of
    each feature for each class instance are calculcated according to bernoulli
    model
    
    Args:
        x_features_train (numpy.2darray): _description_
        y_labels_train (numpy.1darray): _description_
        alpha (int): Dirichlet prior constant. Defaults to 1.

    Returns:
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
    for i, class_instance in enumerate(y_labels_train):
        class_counts[class_instance] += 1
        feature_counts[class_instance] += [1 if x_features_train[i, j] > 0 else 0 for j in range(num_features)]

    prior_probab_calculated = (class_counts) / (len(y_labels_train))
    likelihood_prob_calculated = (feature_counts + alpha) / (class_counts.reshape(-1, 1) + alpha * 2)

    return prior_probab_calculated, likelihood_prob_calculated

# Function to predict
def predict_bernoulli(x_features_test, prior_probabilities, likelihood_probabilites):
    x_binary = (x_features_test > 0).astype(int)
    log_likelihood_prob = np.where(likelihood_prob < 0.0000000001, -10**12, np.log(likelihood_prob,where = likelihood_prob > 0.0000000001))
    log_probs = np.log(prior_prob) + np.dot(x_binary, log_likelihood_prob.T) + np.dot(1 - x_binary, np.log(1 - likelihood_prob).T)
    return np.argmax(log_probs, axis=1)

# Train the Multinomial Naive Bayes model
prior_prob, likelihood_prob = calculate_probabilities_bernoulli(x_train, y_train)
y_pred = predict_bernoulli(x_test, prior_prob, likelihood_prob)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.3f}')

# Create the Confusion Matrix for multinomial bayes with no smoothing
conf_matrix = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))))
for index, label in enumerate(y_test):
    conf_matrix[y_pred[index], y_test[index]] += 1

print('Confusion Matrix:')
print(conf_matrix)
