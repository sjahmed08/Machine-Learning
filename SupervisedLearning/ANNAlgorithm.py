from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import time

def get_ann_results_dataset_1(X_train, y_train, X_test, y_test):
    NeuralNet = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
    NeuralNet.fit(X_train, y_train)
    yPred = NeuralNet.predict(X_test)
    nn_accuracy = accuracy_score(y_test, yPred)

    # Regularization parameter
    alpha_range = np.logspace(-3, 3, 7)
    train_scores, test_scores = validation_curve(NeuralNet, X_train, y_train, param_name="alpha", param_range=alpha_range,
                                                 cv=5,
                                                 n_jobs=4)

    plt.figure()
    plt.semilogx(alpha_range, np.mean(train_scores, axis=1), label='Training score')
    plt.semilogx(alpha_range, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Validation curve for neural network')
    plt.xlabel('alpha (regularization parameter)')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('ann_validation_curve_1.png')
    lr_range = np.logspace(-5, 0, 6)
    train_scores, test_scores = validation_curve(NeuralNet, X_train, y_train, param_name="learning_rate_init",
                                                 param_range=lr_range,
                                                 cv=5, n_jobs=4)
    plt.figure()
    plt.semilogx(lr_range, np.mean(train_scores, axis=1), label='Training score')
    plt.semilogx(lr_range, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Validation curve for neural network')
    plt.xlabel('Learning rate')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('ann_validation_curve_2.png')

    alpha_range = np.logspace(-1, 2, 5)
    lr_range = np.logspace(-5, 0, 6)
    tuned_params = {'alpha': alpha_range, 'learning_rate_init': lr_range}
    NeuralNet = GridSearchCV(NeuralNet, param_grid=tuned_params, cv=5, n_jobs=4)
    start_time = time.time()
    NeuralNet.fit(X_train, y_train)
    end_time = time.time()
    trainTime = end_time - start_time
    print('Training time %f seconds' % trainTime)
    best_NeuralNet = NeuralNet
    best_params = NeuralNet.best_params_
    print("Best parameters set found on development set:")
    print(best_params)

    y_pred = NeuralNet.predict(X_test)
    cv_accuracy = accuracy_score(y_test, y_pred)
    print('CV Accuracy of neural network %.2f%%' % (cv_accuracy * 100))

    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sz, train_scores, test_scores = learning_curve(best_NeuralNet, X_train, y_train, train_sizes=train_sizes, cv=5,
                                                  n_jobs=4)

    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Learning curve for neural network')
    plt.xlabel('Fraction of training examples')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('ann_learning_curve.png')
    return cv_accuracy, trainTime
