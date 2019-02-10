from __future__ import division
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_breast_cancer_data(filename):
    data = pd.read_csv(filename)
    y = data.diagnosis

    drop = ['Unnamed: 32', 'id', 'diagnosis']
    X = data.drop(drop, axis=1)

    # Convert string labels to numerical values
    y = y.values
    y[y == 'M'] = 1
    y[y == 'B'] = 0
    y = y.astype(int)

    print('Total number of examples in the dataset: %d' % X.shape[0])
    positives = 0
    negatives = 0
    for i in range((len(y))):
        if (y[i] == 1):
            positives += 1
        else:
            negatives += 1

    positivePercentage = (positives / X.shape[0])
    negativePercentage = (negatives / X.shape[0])
    print "positive percentage", positivePercentage
    print "negative percentage", negativePercentage

    return X, y

def load_phishing_site_data(filename):
    data = pd.read_csv(filename)

    X = data.values[:, 0: -1]
    y = data.values[:, -1]
    print('Total number of examples in the dataset: %d' % X.shape[0])
    positives = 0
    negatives = 0
    for i in range((len(y))):
        if (y[i] == 1):
            positives += 1
        else:
            negatives += 1

    positivePercentage = (positives / X.shape[0])
    negativePercentage = (negatives / X.shape[0])
    print "positive percentage", positivePercentage
    print "negative percentage", negativePercentage
    return X, y