from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import time

def get_dt_results_dataset_1(X_train, y_train, X_test, y_test):
    dTree = tree.DecisionTreeClassifier(random_state=7)
    dTree.fit(X_train, y_train)
    y_pred = dTree.predict(X_test)
    depth_range = np.arange(30) + 1
    train_scores, test_scores = validation_curve(dTree, X_train, y_train, param_name="max_depth",
                                                 param_range=depth_range, cv=5,
                                                 n_jobs=4)

    plt.figure()
    plt.plot(depth_range, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(depth_range, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Validation curve for decision tree')
    plt.xlabel('max_depth')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('dt_validation_curve_1.png')

    depth_range = np.arange(20) + 1
    tuned_params = {'max_depth': depth_range}
    dTree = GridSearchCV(dTree, param_grid=tuned_params, cv=5, n_jobs=4)
    start_time = time.time()
    dTree.fit(X_train, y_train)
    end_time = time.time()
    trainTime = end_time - start_time
    print('Training time %f seconds' % trainTime)
    best_dTree = dTree
    best_dt_params = dTree.best_params_
    print "Hyper parameter", best_dt_params

    y_pred = dTree.predict(X_test)
    cv_accuracy = accuracy_score(y_test, y_pred)
    print('CV Accuracy of decision tree %.2f%%' % (cv_accuracy * 100))

    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sz, train_scores, test_scores = learning_curve(best_dTree, X_train, y_train, train_sizes=train_sizes, cv=5,
                                                  n_jobs=4)

    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Learning curve for decision tree')
    plt.xlabel('Fraction of training examples')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('dt_learning_curve.png')
    return cv_accuracy, trainTime
