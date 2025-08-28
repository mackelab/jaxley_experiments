

import sys
from jax import config

config.update("jax_enable_x64", True)
#config.update("jax_platform_name", "cpu")

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
import jax.numpy as jnp
import numpy as np
import pickle

import matplotlib.pyplot as plt
import bluepyopt as bpopt
from bluepyopt.parameters import Parameter

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
handler = logging.FileHandler(f'../results/logs/log{now}.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    n_particles=500,
    smc_steps=20,
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
    names, lower, upper = get_bounds()
    
    transform_params, inv_transform_params = transform_uniform_to_normal(lower, upper)
    sample_prior = get_prior(lower, upper, transform_params)

    @jax.jit
    def predict(params):
        return simulate(params)[0]
    
    loss_fn = get_scaled_loss(x_o, predict, 1.0)
    
    logger.info("Running GA...")
    # ga_losses = []

    class HHEvaluator(bpopt.evaluators.Evaluator):
        def __init__(self, x_labels, params):
            super().__init__(objectives=x_labels, params=params)
            self.all_losses = []

        def init_simulator_and_evaluate_with_lists(self, param_list):
            # global ga_losses
            param_list = jnp.asarray(param_list)
            losses = jax.vmap(loss_fn)(param_list)[:, None].tolist()
            logger.info(f"Minimal loss {np.min(losses)}")
            self.all_losses.append(np.min(losses).item())
            return losses

    def map_fun(evaluate_with_lists, list_of_thetas):
        results = evaluate_with_lists(list_of_thetas)
        return results

    x_labels = ["loss"]

    names, lowers, uppers = get_bounds()
    params = []
    for key, lower, upper in zip(names, lowers, uppers):
        params.append(Parameter(f"{key}", bounds=[lower, upper]))

    evaluator = HHEvaluator(x_labels, params)
    opt = bpopt.deapext.optimisations.IBEADEAPOptimisation(
        evaluator,
        offspring_size=n_particles,
        map_function=map_fun,
        seed=seed,
        eta=10,
        mutpb=1.0,
        cxpb=1.0,
    )
    ga_losses = evaluator.all_losses
    final_pop, halloffame, log, hist = opt.run(max_ngen=smc_steps)

    filename = f"ga_250305_{observation}{setup}_{int(t_max)}_{n_particles}_{smc_steps}_{seed}"
    jnp.savez(
        f"../results/{filename}.npz",
        hof=halloffame,
        final_pop=final_pop,
    )
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

    @jax.jit
    def predict(params):
        return simulate(params)[0]

    res = jnp.load(f"../results/{filename}.npz")
    samples = res.get("final_pop")
    samples_hof = res.get("hof")
    # Append last population and HOF.
    samples = jnp.concatenate([samples, samples_hof], axis=0)
    vmapped_predict = jax.vmap(predict)
    x_pred = vmapped_predict(samples[-n_particles-10:])

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
    pred_spikes = jax.vmap(find_spikes, in_axes=(0, None, None, None))(x_pred, None, None, thr)

    pred_ss = {
        "spike_count": np.asarray([spike_count(p) for p in pred_spikes]),
        "first_spike_time": np.asarray([first_spike_time(p) for p in pred_spikes]),
        "second_spike_time": np.asarray([second_spike_time(p) for p in pred_spikes]),
        "first_spike_amp": np.asarray([first_spike_amp(xp, p) for xp, p in zip(x_pred, pred_spikes)]),
        "second_spike_amp": np.asarray([second_spike_amp(xp, p) for xp, p in zip(x_pred, pred_spikes)]),
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
        samples=samples,
        errors=errors,
        summed_errors=summed_errors,
    )
    fig, ax = plt.subplots(1, 1, figsize=(9, 2))
    _ = ax.plot(x_pred[inds[:10]].T, c="gray", alpha=0.2)
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
    parser.add_argument("--loss_fn", type=str, default="dtw_reg")
    parser.add_argument("--n_particles", type=int, default=500)
    parser.add_argument("--smc_steps", type=int, default=20)
    parser.add_argument("--mcmc_steps", type=int, default=1)
    parser.add_argument("--mcmc_step_size", type=float, default=0.2)
    parser.add_argument("--initial_proposal", type=int, default=50_000)
    parser.add_argument("--setup", type=str, default="485574832")
    args = parser.parse_args()
    
    logger.info(f"Start run with seed {args.seed}")
    logger.info(f"Observation: {args.observation}")
    logger.info(f"Loss function: {args.loss_fn}")
    logger.info(f"Number of particles: {args.n_particles}")
    logger.info(f"SMC steps: {args.smc_steps}")
    logger.info(f"MCMC steps: {args.mcmc_steps}")
    logger.info(f"Initial proposal: {args.initial_proposal}")
    logger.info(f"t_max: {args.t_max}")
    logger.info(f"Setup: {args.setup}")
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Device: {jax.devices()}")

    main(
        args.seed,
        args.setup,
        args.observation,
        args.t_max,
        args.n_particles,
        args.smc_steps,
    )