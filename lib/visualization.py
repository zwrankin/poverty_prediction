import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


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
