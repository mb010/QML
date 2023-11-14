import jax
import jax.numpy as jnp
import pennylane as qml
import optax

n_qubits = 8
jax.config.update("jax_enable_x64", True)
dev = qml.device("default.qubit.jax", wires=n_qubits)


# TODO: Check if this can even operate on anything other than graphs? Currently looking at:
# https://docs.pennylane.ai/en/stable/code/qml_qaoa.html
def input_prep():
    """Data encoding circuit."""
    for wire in range(n_qubits):
        qml.Hadamard(wires=wire)


@jax.jit
@qml.qnode(dev, interface="jax", diff_method="backprop")
def model(weights, x):
    qml.QAOAEmbedding(
        features=x,
        weights=weights,
        wires=[w for w in range(n_qubits)],
        local_field="Y",
    )
    return qml.expval(qml.PauliZ(wires=0))


def init_weights(shape, seed=None):
    """Initialises the weights."""
    n_layers = shape[0]
    n_qubits = shape[1]  # must be >= n_features
    shape = qml.QAOAEmbedding.shape(n_layers=n_layers, n_wires=n_qubits)
    weights = jax.random.uniform(
        jax.random.PRNGKey(seed),
        shape=(shape),
        minval=0.0,
        maxval=2 * jnp.pi,
    )
    return weights


def criterion(weights, x, labels, model):
    """Loss function."""
    logits = model(weights, x)
    loss = optax.softmax_cross_entropy(logits, labels)
    return loss
