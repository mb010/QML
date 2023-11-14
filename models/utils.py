import pandas as pd
import pennylane as qml

import optax
import jax.numpy as jnp

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


evaluation_metrics = {
    "Accuracy": sklearn.metrics.accuracy_score,
    "AUC": sklearn.metrics.roc_auc_score,
    "F1Score": sklearn.metrics.f1_score,
    "Precision": sklearn.metrics.precision_score,
    "Recall": sklearn.metrics.recall_score,
}


def criterion(model, x, labels):
    """Loss function."""
    logits = model(x)
    loss = optax.softmax_cross_entropy(logits, labels)
    return loss


def train_test_data(train_size=0.8, seed=None):
    """Loads, formats and splits the pulsar data.

    Returns:
        X_train (np.ndarray): Training data.
        X_test (np.ndarray): Testing data.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
    """
    df = pd.read_csv("pulsar.csv").dropna()
    data = preprocessing.MinMaxScaler(feature_range=(0, jnp.pi)).fit_transform(df)
    X = data[:, :-1]
    y = df["class"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_size, random_state=seed, shuffle=True, stratify=y
    )
    return X_train, X_test, y_train, y_test


def print_model(model, weights) -> str:
    return qml.draw(model)(weights)


def plot_model(model):
    return qml.draw_mpl(model)
