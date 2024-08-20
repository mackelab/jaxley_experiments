from jax import config

config.update("jax_enable_x64", True)
#config.update("jax_platform_name", "cpu")

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import jax.numpy as jnp

import jax


from jaxley_mech.channels.l5pc import *


import logging
import blackjax

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from voltage_imaging_utils import *


def sample_prior(key, initial_position, prior_log_prob, num_steps=2000, verbose=False):
    num_particles = initial_position.shape[0]
    hmc_parameters ={
    'inverse_mass_matrix': jnp.ones((initial_position.shape[-1],)),
    'num_integration_steps': 5,
    'step_size': 0.1} # Larger is kinda better to explore more (even if acceptance rates goes down to 0)
    
    prior_sampling_alg = blackjax.hmc(logdensity_fn=prior_log_prob,**hmc_parameters)
    state_prior = jax.vmap(prior_sampling_alg.init)(initial_position)
    step_fn = jax.jit(jax.vmap(prior_sampling_alg.step))
    

    for i in range(num_steps):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_particles)
        state_prior, info = step_fn(keys, state_prior)
        if verbose:
            print("ll: ", state_prior.logdensity.mean(), "acceptance rate: ", info.acceptance_rate.mean(), "grad norm: ", jnp.linalg.norm(state_prior.logdensity_grad,axis=-1).mean())
    
    return state_prior.position


def sample_posterior(key, initial_position, log_likelihood, prior_log_prob, num_steps=500, step_size=0.01, verbose=False):
    num_particles = initial_position.shape[0]
    num_params = initial_position.shape[-1]
    hmc_parameters ={
    'inverse_mass_matrix': jnp.ones((num_params,)),
    'num_integration_steps': 5,
    'step_size': step_size} # Larger is kinda better to explore more (even if acceptance rates goes down to 0)

    sampling_alg = blackjax.hmc(logdensity_fn=lambda x: prior_log_prob(x) + log_likelihood(x),**hmc_parameters)
    state = jax.vmap(sampling_alg.init)(initial_position)
    step_fn = jax.jit(jax.vmap(sampling_alg.step))

    for i in range(num_steps):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_particles)
        state, info = step_fn(keys, state)
        if verbose:
            print("ll: ", state.logdensity.mean(), "acceptance rate: ", info.acceptance_rate.mean(), "grad norm: ", jnp.linalg.norm(state.logdensity_grad,axis=-1).mean())
        
    return state.position


def main(seed, likelihood_std=0.001, num_steps=200, step_size=0.01):
    logger.info("Building cell")
    print("Building cell")
    cell, num_params, parameters, bounds = build_cell()
    
    logger.info("Setting up the simulation")
    print("Setting up the simulation")
    unflatten, transform = get_transforms(parameters, bounds)
    
    simulate, x_o = get_simulator_and_xo(cell)
    
    loss_from_v, regularizer = get_loss_and_regularizer(cell, x_o)
    
    def sample_proposal(key):
        return jax.random.normal(key, shape=(num_params,)) *2
    
    @jax.jit
    def log_det_transform(params):
        t = lambda x: jax.tree_util.tree_reduce(lambda x,y: jnp.add(x.sum(), y.sum()), transform.forward(x))
        grad = jax.grad(t)(params)
        log_det = jax.tree_util.tree_map(lambda x: jnp.log(jnp.abs(x)), grad)
        log_det_sum = jax.tree_util.tree_reduce(lambda x,y: jnp.add(x.sum(), y.sum()), log_det)
        return log_det_sum
    
    def prior_log_prob(params):
        unflattened_params = unflatten(params)
        return -regularizer(unflattened_params) + log_det_transform(unflattened_params)


    def log_likelihood(params):
        unflattened_params = unflatten(params)
        params = transform.forward(unflattened_params)
        vs = simulate(params)
        return -loss_from_v(vs)/likelihood_std
    
    logger.info("Sampling")
    print("Sampling")
    
    rng_key = jax.random.PRNGKey(seed)
    rng_key, init_key, prior_key,sample_key = jax.random.split(rng_key, 4)
    initial_position = jax.vmap(sample_proposal)(jax.random.split(init_key, 200))
    
    logger.info("Sampling prior")
    print("Sampling prior")
    prior_samples = sample_prior(prior_key, initial_position, prior_log_prob, num_steps=2000, verbose=False)
    logger.info("Sampling posterior")
    print("Sampling")
    posterior_samples = sample_posterior(sample_key, prior_samples, log_likelihood, prior_log_prob, num_steps=num_steps, step_size=step_size,verbose=True)
    
    # Seed 6
    jnp.save(f"md_seed2_new_prior_samples_{str(seed)}_{str(likelihood_std)}.npy", prior_samples)
    jnp.save(f"md_seed2_new_posterior_samples_{str(seed)}_{str(likelihood_std)}.npy", posterior_samples)
    
    

if __name__ == "__main__":
    # Fetch first argument as seed
    import sys
    seed = int(sys.argv[1])
    if len(sys.argv) > 2:
        likelihood_std = float(sys.argv[2])
    else:
        likelihood_std = 0.001
    if len(sys.argv) > 3:
        num_steps = int(sys.argv[3])
    else:
        num_steps = 200
    
    if len(sys.argv) > 4:
        step_size = float(sys.argv[4])
    else:
        step_size = 0.01
    print("Running script")
    main(seed,likelihood_std, num_steps, step_size)
    

