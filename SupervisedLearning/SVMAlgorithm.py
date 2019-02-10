from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import time

def get_svmLinear_results_dataset_1(X_train, y_train, X_test, y_test):
    svm_linear = svm.SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)

    C_range = np.logspace(-3, 3, 7)
    train_scores, test_scores = validation_curve(svm_linear, X_train, y_train, param_name="C", param_range=C_range,
                                                 cv=5, n_jobs=4)

    plt.figure()
    plt.semilogx(C_range, np.mean(train_scores, axis=1), label='Training score')
    plt.semilogx(C_range, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Validation curve for SVM (poly kernel)')
    plt.xlabel('C')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('svm_validation_curve_1.png')

    C_range = np.logspace(-2, 1, 10)
    tuned_params = {'C': C_range}
    svm_linear = GridSearchCV(svm_linear, param_grid=tuned_params, cv=5, n_jobs=4)
    time_start = time.time()
    svm_linear.fit(X_train, y_train)
    time_end = time.time()
    trainTime = time_end - time_start
    print('Training time %f seconds' % trainTime)
    best_clf_svm = svm_linear
    best_params = svm_linear.best_params_
    print "Hyper parameter", best_params
    print(best_params)
    y_pred = best_clf_svm.predict(X_test)

    cv_accuracy = accuracy_score(y_test, y_pred)
    print('CV Accuracy of svm %.2f%%' % (cv_accuracy * 100))

    train_sizes = np.linspace(0.1, 1.0, 5)
    train_size, train_scores, test_scores = learning_curve(best_clf_svm, X_train, y_train, train_sizes=train_sizes, cv=5,
                                                  n_jobs=4)
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Learning curve for SVM (Linear kernel)')
    plt.xlabel('Fraction of training examples')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('svm_learning_curve.png')
    return cv_accuracy, trainTime


def get_svmPoly_results_dataset_1(X_train, y_train, X_test, y_test):
    svm_poly = svm.SVC(kernel='poly')
    svm_poly.fit(X_train, y_train)

    C_range = np.logspace(-3, 3, 7)
    train_scores, test_scores = validation_curve(svm_poly, X_train, y_train, param_name="C", param_range=C_range,
                                                 cv=5, n_jobs=4)

    plt.figure()
    plt.semilogx(C_range, np.mean(train_scores, axis=1), label='Training score')
    plt.semilogx(C_range, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Validation curve for SVM (poly kernel)')
    plt.xlabel('C')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('svm2_validation_curve_1.png')

    C_range = np.logspace(-2, 1, 10)
    tuned_params = {'C': C_range}
    svm_poly = GridSearchCV(svm_poly, param_grid=tuned_params, cv=5, n_jobs=4)
    time_start = time.time()
    svm_poly.fit(X_train, y_train)
    time_end = time.time()
    trainTime = time_end - time_start
    print('Training time %f seconds' % trainTime)
    best_clf_svm = svm_poly
    best_params = svm_poly.best_params_
    print "Hyper parameter", best_params
    y_pred = best_clf_svm.predict(X_test)

    cv_accuracy = accuracy_score(y_test, y_pred)
    print('CV Accuracy of svm %.2f%%' % (cv_accuracy * 100))

    train_sizes = np.linspace(0.1, 1.0, 5)
    train_size, train_scores, test_scores = learning_curve(best_clf_svm, X_train, y_train, train_sizes=train_sizes,
                                                           cv=5,
                                                           n_jobs=4)
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Learning curve for SVM (Poly kernel)')
    plt.xlabel('Fraction of training examples')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('svm2_learning_curve.png')
    return cv_accuracy, trainTime

