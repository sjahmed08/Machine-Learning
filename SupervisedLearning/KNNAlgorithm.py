from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import time

def get_knn_results_dataset_1(X_train, y_train, X_test, y_test):
    k_range = np.arange(1, 101)
    train_scores, test_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, param_name="n_neighbors",
                                                 param_range=k_range, cv=5, n_jobs=4)

    plt.figure()
    plt.plot(k_range, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(k_range, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Validation curve for kNN')
    plt.xlabel('k')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('knn_validation_curve_1.png')

    k_optimal = np.argmax(np.mean(test_scores, axis=1)) + 1
    print('Optimal value of k: %d' % k_optimal)
    best_clf_knn = KNeighborsClassifier(n_neighbors=k_optimal)
    start_time = time.time()
    best_clf_knn.fit(X_train, y_train)
    end_time = time.time()
    trainTime = end_time - start_time
    print('Training time %f seconds' % trainTime)
    y_pred = best_clf_knn.predict(X_test)
    cv_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of kNN with k = %d is %.2f%%' % (k_optimal, cv_accuracy * 100))

    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sz, train_scores, test_scores = learning_curve(best_clf_knn, X_train, y_train, train_sizes=train_sizes, cv=5,
                                                  n_jobs=4)

    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Learning curve for kNN')
    plt.xlabel('Fraction of training examples')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('knn_learning_curve.png')
    return cv_accuracy, trainTime