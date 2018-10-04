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


def plot_feature_importances(df, n=20, threshold=None, return_df=False):
    """
    Slightly adapted from https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough
    Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances.

    Args:
        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".

        n (int): Number of most important features to plot. Default is 15.

        threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made. Default is None.

    Returns:
        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1)
                        and a cumulative importance column

    Note:

        * Normalization in this case means sums to 1.
        * Cumulative importance is calculated by summing features from most to least important
        * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance

    """
    plt.style.use('fivethirtyeight')

    # Sort features with most important at the head
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    # Normalize the feature importances to add up to one and calculate cumulative importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    plt.rcParams['font.size'] = 12

    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y='importance_normalized',
                            x='feature', color='darkgreen',
                            edgecolor='k', figsize=(12, 8),
                            legend=False, linewidth=2)

    plt.xlabel('Normalized Importance', size=18);
    plt.ylabel('');
    plt.title(f'{n} Most Important Features', size=18)
    plt.gca().invert_yaxis()

    if threshold:
        # Cumulative importance plot
        plt.figure(figsize=(8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size=16);
        plt.ylabel('Cumulative Importance', size=16);
        plt.title('Cumulative Feature Importance', size=18);

        # Number of features needed for threshold cumulative importance
        # This is the index (will need to add 1 for the actual number)
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))

        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin=0, ymax=1.05, linestyles='--', colors='red')
        plt.show();

        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1,
                                                                                  100 * threshold))

    if return_df:
        return df
