# Adapted from: https://pennylane.ai/qml/demos/tutorial_contextuality/
# And recreating equivalent model to Kordzanganeh+2021: https://arxiv.org/abs/2112.02655
import numpy as np
import pandas as pd
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax


import models.utils as model_utils

jax.config.update("jax_enable_x64", True)


def training_loop(
    model,
    weights,
    criterion,
    nstep=100,
    lr=0.01,
    log_metrics=1,
    seed=42,
    train_frac=1.0,
    evaluation_metrics=None,
):
    """Optimises the model.

    Args:
        model (qnode): Quantum circuit.
        weights (jnp.ndarray): Initial weights.
        criterion (func): Loss function.
        nstep (int, optional):
            Number of steps / epochs. Defaults to 100.
        lr (float, optional):
            Learning rate. Defaults to 0.01.
        depth (int, optional):
            Depth of the circuit. Defaults to 3.
        seed (int, optional):
            Random seed. Defaults to 42.
        train_frac (float, optional):
            Fraction of training data to use (not stratified). Should be in range [0., 1.].
            Defaults to 1.0.
        evaluation_metrics (dict, optional):
            Dictionary of evaluation metrics for logging with each
            value being a function. Defaults to None.

    Returns:
        weights (np.ndarray): Optimised weights.
        metrics (pd.DataFrame): Metrics.
    """

    # Load data.
    assert train_frac <= 1.0, "train_frac must be <= 1.0"
    X_train, X_test, y_train, y_test = model_utils.train_test_data(
        train_size=train_frac * 0.8, seed=seed
    )
    X_train = jnp.asarray(X_train[: int(X_train.shape[0])])
    y_train = jnp.asarray(y_train[: int(y_train.shape[0])])
    X_test = jnp.asarray(X_test)
    y_test = jnp.asarray(y_test)

    # Initialise metric dict.
    metrics = {
        "epoch": [],
        "loss": [],
    }
    if evaluation_metrics:
        for name in evaluation_metrics.keys():
            metrics[name] = []

    # Initialise optimizer.
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(weights)

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


def configure_model(
    model_name, vmapped=False, seed=None, n_layers=3, n_features=8, n_qubits=8
):
    if model_name == "QUAM":
        shape = (n_layers + 1, n_features, 2)
        import models.quam as model_module
    elif model_name == "QAOA":
        shape = [n_layers, n_qubits]
        import models.qaoa as model_module
    else:
        raise NotImplementedError

    model = model_module.model
    if vmapped:
        model = jax.vmap(model, in_axes=(None, 1))
    weights = model_module.init_weights(shape, seed=seed)
    criterion = model_module.criterion

    return model, weights, criterion


def quick_train(model_name, vmapped=False, seed=42):
    model, weights, criterion = configure_model(
        model_name, vmapped=vmapped, seed=seed, n_layers=3, n_features=8
    )

    weights, metrics = training_loop(
        model,
        weights,
        criterion,
        nstep=100,
        lr=0.01,
        log_metrics=1,
        seed=seed,
        train_frac=1.0,
        evaluation_metrics=model_utils.evaluation_metrics,
    )

    metrics.to_csv(f"results/{model_name.lower()}_quick_metrics.csv")
    np.save(f"results/{model_name.lower()}_quick_weights.npy", weights)


def sweep_train(model_name, vmapped=False, seed=42):
    initialised = False
    for seed in range(0, 5):
        for idx, depth in enumerate(range(1, 10)):
            for train_frac in np.linspace(0.5, 1.0, 3, endpoint=True):
                model, weights, criterion = configure_model(
                    model_name, vmapped=vmapped, seed=seed, n_layers=depth, n_features=8
                )
                weights, df = training_loop(
                    model,
                    weights,
                    criterion,
                    nstep=1000,
                    lr=0.01,
                    log_metrics=1,
                    seed=seed,
                    train_frac=train_frac,
                    evaluation_metrics=model_utils.evaluation_metrics,
                )
                if initialised:
                    df_all = pd.concat([df_all, df])
                    sweep_weights.append(np.asarray(weights))
                else:
                    df_all = df
                    sweep_weights = [weights]
                    initialised = True

    df_all.to_csv(f"results/{model_name.lower()}_sweep_metrics.csv")
    np.save(
        f"results/{model_name.lower()}_sweep_weights.npy",
        np.concatenate(sweep_weights, axis=0),
    )


if __name__ == "__main__":
    # sweep_train("QUAM", vmapped=False, seed=42)
    # import timeit
    # timeit.timeit(lambda: quick_train("QUAM", vmapped=False, seed=42), number=3)
    # timeit.timeit(lambda: quick_train("QUAM", vmapped=True, seed=42), number=3)
    quick_train("QUAM", vmapped=True, seed=42)
    # quick_train("QAOA", vmapped=False, seed=42)
