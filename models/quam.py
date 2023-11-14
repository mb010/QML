import jax
import jax.numpy as jnp
import pennylane as qml
import optax


jax.config.update("jax_enable_x64", True)
dev = qml.device("default.qubit.jax", wires=1)


def input_prep():
    """Data encoding circuit."""
    qml.Hadamard(wires=0)


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


def init_weights(shape, seed=None):
    """Initialises the weights."""
    weights = jax.random.uniform(
        jax.random.PRNGKey(seed),
        shape=shape,
        minval=0.0,
        maxval=2 * jnp.pi,
    )
    return weights


def criterion(weights, x, labels, model):
    """Loss function."""
    logits = model(weights, x)
    loss = optax.softmax_cross_entropy(logits, labels)
    return loss
