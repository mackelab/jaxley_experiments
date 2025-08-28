import sys
from jax import config

config.update("jax_enable_x64", True)

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import optax
from copy import deepcopy

from nex.allen_fits.build_simulator import (
    get_bounds,
    setup_simulator,
    get_prior,
    get_experimental_data,
    build_cell,
    set_setup,
    transform_uniform_to_normal,
)
from nex.allen_fits.loss_util import (
    soft_dtw_distance,
    window_reduce,
)
from nex.allen_fits.posthoc_summary_stats import (
    find_spikes,
    spike_count,
    first_spike_time,
    second_spike_time,
    first_spike_amp,
    second_spike_amp,
)


from warnings import simplefilter
import pandas as pd

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


import logging
import datetime

now = datetime.datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger(__name__)

# Create a file handler
handler = logging.FileHandler(f"../results/logs/log{now}.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Create a stream handler to output to console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


def get_scaled_loss(x_o, predict, cost_fn_power, **kwargs):
    gamma = 0.1

    x_o_red = window_reduce(x_o, jnp.max, stride=30, window_size=50)
    min_x = jnp.min(x_o_red)
    max_x = jnp.max(x_o_red)
    x_o_red = (x_o_red - min_x) / (max_x - min_x)

    def cost_fn(x, y):
        c = (jnp.abs(x - y)) ** cost_fn_power
        return c

    def penalty_fn(rows, cols):
        return jnp.abs((rows - cols) / x_o_red.shape[0])

    c1 = soft_dtw_distance(
        x_o_red, x_o_red, gamma=gamma, cost_fn=cost_fn, penalty_fn=penalty_fn
    )

    @jax.jit
    def loss_fn(x):
        x = window_reduce(x, jnp.max, stride=30, window_size=50)
        x = (x - min_x) / (max_x - min_x)
        l1 = soft_dtw_distance(
            x, x_o_red, gamma=gamma, cost_fn=cost_fn, penalty_fn=penalty_fn
        )
        l2 = soft_dtw_distance(
            x, x, gamma=gamma, cost_fn=cost_fn, penalty_fn=penalty_fn
        )
        return l1 - 0.5 * l2 - 0.5 * c1

    @jax.jit
    def scaled_loss(params, scale=0.05):
        x = predict(params)
        return loss_fn(x) / scale

    return scaled_loss


def main(
    seed,
    setup,
    observation,
    t_max,
    steps=20,
    momentum=0.0,
    cost_fn_power=1.0,
):
    logger.info(f"Start run with seed {seed}")
    logger.info("Building cell and loading data...")
    logger.info(f"Observation: {observation}")
    logger.info(f"Setup: {setup}")
    set_setup(setup, t_max)
    cell = build_cell()
    t, x_o = get_experimental_data(t_max)
    simulate = setup_simulator(cell, t_max)
    x_o = x_o.astype(jnp.float64)

    logger.info("Building prior and log likelihood function...")
    _, lower, upper = get_bounds()

    transform_params, inv_transform_params = transform_uniform_to_normal(lower, upper)
    sample_prior = get_prior(lower, upper, transform_params)

    @jax.jit
    def predict(params):
        params_0 = inv_transform_params(params)
        return simulate(params_0)[0]

    scaled_loss = get_scaled_loss(x_o, predict, cost_fn_power)
    grad_fn = jax.jit(jax.vmap(jax.value_and_grad(scaled_loss)))

    logger.info("Running Gradient Descent...")

    rng_key = jax.random.PRNGKey(seed)
    init_rng_key, sampling_rng_key = jax.random.split(rng_key)

    # Sample 1000 initializations randomly.
    def sample_prior(key):
        u = jax.random.uniform(key, shape=lower.shape, minval=lower, maxval=upper)
        u = transform_params(u)
        return u

    n_seeds = 10
    n_lrs = 1
    n_particles = n_seeds * n_lrs
    opt_params = jax.vmap(sample_prior)(
        jax.random.split(init_rng_key, n_seeds)
    )
    logger.info(f"opt_params.shape {opt_params.shape}")

    opt_params = jnp.tile(opt_params, (n_lrs, 1))
    # lrs = np.logspace(-5, -1, 100)
    # lrs = lrs[40:60]
    # lrs = lrs[4:5]  # 0.00059948
    lrs = np.asarray([5e-4])
    lrs = np.repeat(lrs, n_seeds)

    # lr=1.0 is not used because we override the learning rate later at
    # `opt_state.hyperparams`.
    optimizer = optax.inject_hyperparams(optax.sgd)(
        learning_rate=1.0, momentum=momentum
    )
    opt_state = jax.vmap(optimizer.init)(opt_params)

    best_loss_per_chain = 1e10 * jnp.ones((n_particles))
    best_opt_params_per_chain = opt_params

    all_losses_each_chain = []

    for epoch in range(steps):
        loss_val, grad_val = grad_fn(opt_params)

        # Save best values per chain.
        condition = loss_val < best_loss_per_chain
        best_loss_per_chain = best_loss_per_chain.at[condition].set(loss_val[condition])
        best_opt_params_per_chain = best_opt_params_per_chain.at[condition].set(
            opt_params[condition]
        )
        all_losses_each_chain.append(deepcopy(best_loss_per_chain))

        grad_norms = jnp.sum(grad_val**2, axis=1) ** 0.5
        logger.info(
            f"Epoch {epoch} Loss: Avg {jnp.nanmean(loss_val)}, Med {jnp.nanmedian(loss_val)}, Min {jnp.nanmin(loss_val)}"
        )

        # Normalize the gradient to (almost exactly) unit length.
        beta = 0.99
        grad_val = (grad_val.T / (grad_norms**beta)).T
        opt_state.hyperparams["learning_rate"] = loss_val * lrs
        updates, opt_state = jax.vmap(optimizer.update)(grad_val, opt_state)
        opt_params = jax.vmap(optax.apply_updates)(opt_params, updates)

    sorting = np.argsort(best_loss_per_chain)
    logger.info(f"10 best losses: {best_loss_per_chain[sorting[:10]]}")

    # Save.
    filename = f"gd_low_250315_{setup}_{int(t_max)}_{momentum}_{steps}_{cost_fn_power}_{seed}"

    particles = best_opt_params_per_chain
    weights = -best_loss_per_chain
    jnp.savez(
        f"../results/{filename}.npz",
        particles=particles,
        weights=weights,
        all_losses_each_chain=all_losses_each_chain
    )

    ############################
    # Group the runs and take the median.
    best_opt_params_per_chain = np.reshape(best_opt_params_per_chain, (n_lrs, n_seeds, -1))
    best_loss_per_chain = np.reshape(best_loss_per_chain, (n_lrs, n_seeds))

    loss_of_lrs = []
    opt_params_at_lrs = []
    for lr in range(n_lrs):
        loss_sorting = np.argsort(best_loss_per_chain[lr])
        median_index = loss_sorting[0]
        opt_params_at_lrs.append(best_opt_params_per_chain[lr, median_index])
        loss_of_lrs.append(best_loss_per_chain[lr, median_index])

    opt_params_at_lrs = jnp.asarray(opt_params_at_lrs)
    loss_of_lrs = jnp.asarray(loss_of_lrs)

    # print("loss_of_lrs", loss_of_lrs)
    # print("opt_params_at_lrs", opt_params_at_lrs[:, 0])

    # To avoid any hiccups.
    best_opt_params_per_chain = opt_params_at_lrs
    best_loss_per_chain = loss_of_lrs

    with open(f"../results/losses/{filename}.pkl", "wb") as handle:
        pickle.dump(loss_of_lrs, handle)
    with open(f"../results/losses/median_opt_params_{filename}.pkl", "wb") as handle:
        pickle.dump(best_opt_params_per_chain, handle)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    _ = ax.plot(loss_of_lrs, marker="o")
    plt.savefig(f"../results/losses/loss_curve_{filename}.png", dpi=200, bbox_inches="tight")

    assert opt_params_at_lrs.shape[0] == n_lrs

    ####################################################################################
    ####################################################################################
    ################################# POSTHOC ANALYSIS #################################
    ####################################################################################
    ####################################################################################

    t_max = 1100.0
    set_setup(setup, t_max)
    cell = build_cell()
    t, x_o = get_experimental_data(t_max)
    simulate = setup_simulator(cell, t_max)
    x_o = x_o.astype(jnp.float64)

    transform_params, inv_transform_params = transform_uniform_to_normal(lower, upper)
    sample_prior = get_prior(lower, upper, transform_params)

    @jax.jit
    def predict(params):
        params_0 = inv_transform_params(params)
        return simulate(params_0)[0]

    vmapped_predict = jax.vmap(predict)
    x_pred = vmapped_predict(best_opt_params_per_chain)

    if setup == "473601979":
        thr = -30.0
    else:
        thr = -20.0  # Default value.

    _ = np.random.seed(0)
    noise = np.random.randn(len(x_o)) * 1e-2
    x_o_spikes = find_spikes(x_o + noise, None, None, thr)

    x_o_ss = {
        "spike_count": spike_count(x_o_spikes),
        "first_spike_time": first_spike_time(x_o_spikes),
        "second_spike_time": second_spike_time(x_o_spikes),
        "first_spike_amp": first_spike_amp(x_o, x_o_spikes),
        "second_spike_amp": second_spike_amp(x_o, x_o_spikes),
    }
    pred_spikes = jax.vmap(find_spikes, in_axes=(0, None, None, None))(
        x_pred, None, None, thr
    )

    pred_ss = {
        "spike_count": np.asarray([spike_count(p) for p in pred_spikes]),
        "first_spike_time": np.asarray([first_spike_time(p) for p in pred_spikes]),
        "second_spike_time": np.asarray([second_spike_time(p) for p in pred_spikes]),
        "first_spike_amp": np.asarray(
            [first_spike_amp(xp, p) for xp, p in zip(x_pred, pred_spikes)]
        ),
        "second_spike_amp": np.asarray(
            [second_spike_amp(xp, p) for xp, p in zip(x_pred, pred_spikes)]
        ),
    }
    errors = {}
    for key in x_o_ss.keys():
        errors[key] = np.abs(x_o_ss[key] - pred_ss[key])

    # Weigh the individusal errors.
    summed_errors = (
        np.asarray(errors["spike_count"]).astype(float) * 5.0
        + np.asarray(errors["first_spike_time"]).astype(float) * 2.0
        + np.asarray(errors["second_spike_time"]).astype(float) * 2.0
        + np.asarray(errors["first_spike_amp"]).astype(float) * 1.0
        + np.asarray(errors["second_spike_amp"]).astype(float) * 1.0
    )

    inds = np.argsort(summed_errors)

    logger.info(f"Minimal error: {summed_errors[inds[0]]:.4f}")
    logger.info("===========")
    for key in errors.keys():
        logger.info(f"Error in {key}: {errors[key][inds[0]]:.4f}")

    # Save the single best trace (traces are memory intensive...)
    with open(f"../results/pickled_traces/{filename}.pkl", "wb") as handle:
        pickle.dump(x_pred[inds[0]], handle)

    # Save the final samples and errors to easily reconstruct which one was best.
    jnp.savez(
        f"../results/resampled_samples/{filename}.npz",
        errors=errors,
        summed_errors=summed_errors,
    )
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    for i, x_plot in enumerate(x_pred[inds[:10]]):
        _ = ax.plot(x_plot + 100.0 * i, c="gray", alpha=0.8)
    _ = ax.plot(x_o, c="k")
    _ = ax.plot(x_pred[inds[0]], c="b")
    _ = ax.set_title(f"Total error {summed_errors[inds[0]]:.4f}")
    plt.savefig(f"../results/pngs/{filename}.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observation", type=str, default="allen")
    parser.add_argument("--t_max", type=float, default=199.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--cost_fn_power", type=float, default=1.0)
    parser.add_argument("--setup", type=str, default="485574832")
    args = parser.parse_args()

    logger.info(f"Start run with seed {args.seed}")
    logger.info(f"Observation: {args.observation}")
    logger.info(f"Steps: {args.steps}")
    logger.info(f"Momentum: {args.momentum}")
    logger.info(f"Cost fn power: {args.cost_fn_power}")
    logger.info(f"t_max: {args.t_max}")
    logger.info(f"Setup: {args.setup}")
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Device: {jax.devices()}")

    main(
        args.seed,
        args.setup,
        args.observation,
        args.t_max,
        args.steps,
        args.momentum,
        args.cost_fn_power,
    )
