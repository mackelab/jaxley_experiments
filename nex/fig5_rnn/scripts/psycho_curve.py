import jax.numpy as jnp
from jax import value_and_grad, jit
import optax


def Weibull(ll_params, x):
    return 1 - jnp.exp(-1 * (x / ll_params["lam"]) ** ll_params["k"])


def logistic_phi(ll_params, x):
    logistic = 1 / (1 + jnp.exp(-1 * ll_params["a"] * (x - ll_params["b"])))
    phi = ll_params["gamma"] + (1 - ll_params["gamma"] - ll_params["lambda"]) * logistic
    return phi


def logistic(ll_params, x):
    """The one used!"""
    return 1 / (1 + jnp.exp(-1 * ll_params["a"] * (x - ll_params["b"])))


def loss_fn(ll_params, stim_means, fracs):
    """MSE"""
    pred = logistic(ll_params, stim_means)
    return jnp.mean((pred - fracs) ** 2)


grad_fn = jit(value_and_grad(loss_fn, argnums=0))


def fit_curve(mean_range, frac_pos):
    # Fit the psychometric curve to the dist
    psycho_params = {"a": 500.0, "b": 0.0}  # init values
    optimizer = optax.adam(learning_rate=0.005)
    opt_state = optimizer.init(psycho_params)

    n_epochs = 200_000

    for i in range(n_epochs):
        l, g = grad_fn(psycho_params, mean_range, frac_pos)

        updates, opt_state = optimizer.update(g, opt_state)
        psycho_params = optax.apply_updates(psycho_params, updates)

        if i % 10_000 == 0:
            print(f"it {i/n_epochs}, loss {l}")
            # print(f"a: {psycho_params['a']}, b: {psycho_params['b']}")

    print(psycho_params)
    return psycho_params
