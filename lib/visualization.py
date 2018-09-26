import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning


def plot_confusion_matrix(y, y_pred, normalize=True, **kwargs):

    conf_mx = confusion_matrix(y, y_pred)

    if normalize:
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        conf_mx = conf_mx / row_sums

    ax = plt.axes()
    sns.heatmap(conf_mx, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, **kwargs)
    ax.set_title('Confusion matrix')
    ax.set_ylabel('Observed')
    ax.set_xlabel('Predicted')


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, scoring=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def cv_model(transformer_pipeline, alg, X_train, y_train, cv, scoring, model_results=None):
    alg_name = alg.__name__
    X = transformer_pipeline.fit_transform(X_train)
    y = y_train

    cv_scores = cross_val_score(alg(), X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f'{alg_name} CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')

    if model_results is not None:
        model_results = model_results.append(pd.DataFrame({'model': alg_name,
                                                           'cv_mean': cv_scores.mean(),
                                                           'cv_std': cv_scores.std()},
                                                          index=[0]),
                                             ignore_index=True)

        return model_results


def report_cv_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def compare_algorithm_cv_scores(transformer_pipeline, algs, X_train, y_train, cv, scoring):
    # Filter out warnings from models
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    model_results = pd.DataFrame(columns=['model', 'cv_mean', 'cv_std'])

    for alg in algs:
        model_results = cv_model(transformer_pipeline, alg, X_train, y_train, cv, scoring, model_results)

    return model_results

def plot_algorithm_cv_scores(model_results):
    model_results.set_index('model', inplace=True)
    model_results['cv_mean'].plot.bar(color='orange', figsize=(8, 6),
                                      yerr=list(model_results['cv_std']),
                                      edgecolor='k', linewidth=2)
    plt.title('Model F1 Score Results');
    plt.ylabel('Mean F1 Score (with error bar)');
    model_results.reset_index(inplace=True)
