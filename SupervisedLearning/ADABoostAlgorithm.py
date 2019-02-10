from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import time

def get_adaboost_results_dataset_1(X_train, y_train, X_test, y_test):
    dTree = tree.DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    dTeeBoosted = AdaBoostClassifier(base_estimator=dTree, random_state=7)
    dTeeBoosted.fit(X_train, y_train)

    y_pred = dTeeBoosted.predict(X_test)
    boosted_accuracy = accuracy_score(y_test, y_pred)

    num_learners = 1500
    # dt_stump = tree.DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    dTeeBoosted = AdaBoostClassifier(base_estimator=dTree, n_estimators=num_learners, random_state=7)

    num_folds = 5
    kf = KFold(n_splits=num_folds, random_state=7)
    train_scores = np.zeros((num_learners, num_folds))
    val_scores = np.zeros((num_learners, num_folds))
    for idx, (train_index, test_index) in enumerate(kf.split(X_train)):
        dTeeBoosted.fit(X_train[train_index], y_train[train_index])
        train_scores[:, idx] = np.asarray(list(dTeeBoosted.staged_score(X_train[train_index], y_train[train_index])))
        val_scores[:, idx] = np.asarray(list(dTeeBoosted.staged_score(X_train[test_index], y_train[test_index])))

    n_estimators_range = np.arange(num_learners) + 1
    plt.figure()
    plt.plot(n_estimators_range, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(n_estimators_range, np.mean(val_scores, axis=1), label='Cross-validation score')
    plt.title('Cross-validation curve for AdaBoost')
    plt.xlabel('Number of weak learners')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('ADA_validation_curve_1.png')

    num_learners_optimal = np.argmax(np.mean(val_scores, axis=1)) + 1
    print('Optimal number of learners for AdaBoost: %d' % num_learners_optimal)
    best_dTeeBoosted = AdaBoostClassifier(base_estimator=dTree, n_estimators=num_learners_optimal, random_state=7)
    time_start= time.time()
    best_dTeeBoosted.fit(X_train, y_train)
    time_end = time.time()
    trainTime = time_end - time_start
    print('Training time %f seconds' % trainTime)
    y_pred = best_dTeeBoosted.predict(X_test)

    cv_accuracy = accuracy_score(y_test, y_pred)
    print('CV Accuracy of adaboost %.2f%%' % (cv_accuracy * 100))

    train_sizes = np.linspace(0.1, 1.0, 5)
    best_dTeeBoosted = AdaBoostClassifier(base_estimator=dTree, n_estimators=num_learners_optimal, random_state=7)
    _, train_scores, test_scores = learning_curve(best_dTeeBoosted, X_train, y_train, train_sizes=train_sizes, cv=5,
                                                  n_jobs=4)

    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Learning curve for AdaBoost')
    plt.xlabel('Fraction of training examples')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('ADA_learning_curve.png')
    return cv_accuracy, trainTime