from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import time

from loadData import *
from SVMAlgorithm import *
from KNNAlgorithm import *
from ADABoostAlgorithm import *
from ANNAlgorithm import *
from DTAlgorithm import *



if __name__ == "__main__":
    X, y = load_breast_cancer_data('breastCancer.csv')
    # X, y = load_phishing_site_data('phishing.csv')
    X = preprocessing.scale(X)
    # print y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=18)

    cvAccuracy = np.zeros(6)
    trainTime = np.zeros(6)
    testTime = np.zeros(6)

    cvAccuracy[0], trainTime[0] = get_svmLinear_results_dataset_1(X_train, y_train, X_test, y_test)
    cvAccuracy[1], trainTime[1] = get_svmPoly_results_dataset_1(X_train, y_train, X_test, y_test)
    cvAccuracy[2], trainTime[2] = get_knn_results_dataset_1(X_train, y_train, X_test, y_test)
    cvAccuracy[3], trainTime[3] = get_adaboost_results_dataset_1(X_train, y_train, X_test, y_test)
    cvAccuracy[4], trainTime[4] = get_dt_results_dataset_1(X_train, y_train, X_test, y_test)
    cvAccuracy[5], trainTime[5] = get_ann_results_dataset_1(X_train, y_train, X_test, y_test)

    print cvAccuracy, trainTime, testTime


