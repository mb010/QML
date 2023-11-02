# Adapted from: https://pennylane.ai/qml/demos/tutorial_contextuality/
# And recreating equivalent model to Kordzanganeh+2021: https://arxiv.org/abs/2112.02655
import numpy as np
import pandas as pd
from tqdm import tqdm

import pennylane as qml

import jax
import jax.numpy as jnp
import optax

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


jax.config.update("jax_enable_x64", True)
SEED = 42


def train_test_data(seed=None):
    """Loads, formats and splits the pulsar data.

    Returns:
        X_train (np.ndarray): Training data.
        X_test (np.ndarray): Testing data.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
    """
    df = pd.read_csv("pulsar.csv").dropna()
    data = preprocessing.MinMaxScaler(feature_range=(0, np.pi)).fit_transform(df)
    # data = jnp.array(data)
    X = data[:, :-1]
    y = df["class"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, shuffle=True, stratify=y
    )
    return X_train, X_test, y_train, y_test


# Has to be defined in the file for the decorators to work.
dev = qml.device("default.qubit.jax", wires=1)


def input_prep():
    """Data encoding circuit."""
    qml.Hadamard(wires=0)


def variational_circuit(weights, layer, feature_n, x=0):
    """Variational circuit."""
    qml.RZ(weights[layer][feature_n][0] + x, wires=0)
    qml.RY(weights[layer][feature_n][1], wires=0)


@jax.jit
@qml.qnode(dev, interface="jax", diff_method="backprop")
def model(weights, x):
    """Quantum circuit."""
    input_prep()
    depth = weights.shape[1]
    qml.RY(weights[0][0][0], wires=0)
    for layer in range(1, depth):
        for feature_n in range(x.shape[0]):
            qml.RZ(weights[layer][feature_n][0] + x[feature_n], wires=0)
            qml.RY(weights[layer][feature_n][1], wires=0)

    return qml.expval(qml.PauliZ(wires=0))


def criterion(weights, x, labels, model):
    """Loss function."""
    logits = model(weights, x)
    # print(logits.shape, labels.shape)
    loss = optax.softmax_cross_entropy(logits, labels)
    return loss


def optimise_model(
    model,
    criterion,
    nstep=100,
    lr=0.01,
    depth=3,
    log_metrics=1,
    seed=42,
    vmap=False,
    train_frac=1.0,
    evaluation_metrics=None,
):
    """Optimises the model.

    Args:
        model (qnode): Quantum circuit.
        criterion (func): Loss function.
        nstep (int, optional):
            Number of steps / epochs. Defaults to 100.
        lr (float, optional):
            Learning rate. Defaults to 0.01.
        depth (int, optional):
            Depth of the circuit. Defaults to 3.
        seed (int, optional):
            Random seed. Defaults to 42.
        vmap (bool, optional):
            Whether to parallelise the model. Defaults to False.
        train_frac (float, optional):
            Fraction of training data to use (not stratified).
            Defaults to 1.0.
        evaluation_metrics (dict, optional):
            Dictionary of evaluation metrics for logging with each
            value being a function. Defaults to None.

    Returns:
        weights (np.ndarray): Optimised weights.
        metrics (pd.DataFrame): Metrics.
    """

    # Load data.
    X_train, X_test, y_train, y_test = train_test_data(seed=seed)
    X_train = jnp.asarray(X_train[: int(X_train.shape[0] * train_frac)])
    y_train = jnp.asarray(y_train[: int(y_train.shape[0] * train_frac)])
    X_test = jnp.asarray(X_test)
    y_test = jnp.asarray(y_test)

    # Initialise metric dict.
    metrics = {
        "epoch": [],
        "loss": [],
        "depth": [depth for _ in range(nstep)],
        "training samples": [train_frac for _ in range(nstep)],
        "seed": [seed for _ in range(nstep)],
    }
    if evaluation_metrics:
        for name in evaluation_metrics.keys():
            metrics[name] = []

    # Initialise weights.
    weights = jax.random.uniform(
        jax.random.PRNGKey(seed),
        shape=(depth + 1, X_train.shape[-1], 2),
        minval=0.0,
        maxval=2 * np.pi,
    )

    # Initialise optimizer.
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(weights)

    # Parallelise the model.
    if vmap:
        model = jax.vmap(model, in_axes=(None, 1))

    # Train the model.
    steps = tqdm(range(nstep))
    for step in steps:
        # Retrieve one hot encoded labels
        labels = jax.nn.one_hot(y_train, 2)[:, 1]

        # Retrieve loss and gradients
        loss, grads = jax.value_and_grad(criterion)(weights, X_train.T, labels, model)

        # Update weights
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)

        # Log metrics
        train_accuracy = jnp.mean(
            jnp.where(model(weights, X_train.T) > 0.5, 1, 0) == y_train
        )
        steps.set_description(
            f"Current loss: {loss:.4f} :::: Current train_accuracy: {train_accuracy:.4f}"
        )
        if step % log_metrics == 0:
            if evaluation_metrics:
                y_pred = jnp.where(model(weights, X_test.T) > 0.5, 1, 0)
                for name, func in evaluation_metrics.items():
                    metrics[name].append(func(y_pred, y_test))

            metrics["epoch"].append(step)
            metrics["loss"].append(float(loss))

    return weights, pd.DataFrame.from_dict(metrics)


evaluation_metrics = {
    "Accuracy": sklearn.metrics.accuracy_score,
    "AUC": sklearn.metrics.roc_auc_score,
    "F1Score": sklearn.metrics.f1_score,
    "Precision": sklearn.metrics.precision_score,
    "Recall": sklearn.metrics.recall_score,
}


def sweep_train():
    for seed in range(10):
        for idx, depth in enumerate(range(1, 10)):
            for train_frac in np.linspace(0.1, 1.0, 10):
                weights, df = optimise_model(
                    model,
                    criterion,
                    nstep=1000,
                    lr=0.01,
                    depth=depth,
                    log_metrics=1,
                    seed=seed,
                    vmap=True,
                    train_frac=train_frac,
                    evaluation_metrics=evaluation_metrics,
                )
            if idx == 0 and seed == 0:
                df_all = df
                sweep_weights = [weights]
            else:
                df_all = pd.concat([df_all, df])
                sweep_weights.append(np.asarray(weights))

    df_all.to_csv(f"results/quam_sweep_metrics.csv")
    np.save(f"results/quam_sweep_weights.npy", np.concat(weights_all))


def quick_train():
    weights, metrics = optimise_model(
        model,
        criterion,
        nstep=100,
        lr=0.01,
        depth=9,
        log_metrics=1,
        seed=42,
        vmap=True,
        train_frac=1.0,
        evaluation_metrics=evaluation_metrics,
    )

    metrics.to_csv("results/quam_quick_metrics.csv")
    np.save("results/quam_quick_weights.npy", weights)


if __name__ == "__main__":
    sweep_train()
