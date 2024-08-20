from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import numpy as np
from tqdm import tqdm
import time as time_module
import pickle
import cProfile
import io
import pstats

import wandb
import yaml
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
from jax.tree_util import tree_map
import optax

import jaxley as jx
from jaxley.optimize.transforms import ParamTransform
from jaxley.optimize.utils import l2_norm

from nex.fig5_rnn.scripts.network import initialize_RNN
from nex.fig5_runn.scripts.EI_stimulus import gen_stim_batch
from nex.fig5_runn.scripts.EI_train_utils import get_synapse_indices

wandb.login()

def softmax(x):
    exp_x = jnp.exp(x - jnp.max(x))
    return exp_x / jnp.sum(exp_x)

def train(config, save_dir, time_stamp, sync_wandb=False, freeze_outs=True):
            
    RNN_params = config["RNN_params"]
    train_params = config["train_params"]
    task_params = config["task_params"]

    # Do the scaling of the initial gain
    g_scaling = 100_000 / 2 / np.pi / 10.0 / 1.0
    RNN_params["init_gain"] /= g_scaling

    network, conn_matrix, init_input_weights = initialize_RNN(RNN_params)
    init_input_weights = init_input_weights[0]["input_weights"]

    network.delete_recordings()

    # Record from the soma of each cell
    n_cells = RNN_params["n_rec"]
    n_readouts = RNN_params["n_out"]
    for i in range(n_cells):
        network.cell([i]).branch(1).comp(0).record(verbose=False)
    # Record from the readout (only one branch)
    for i in range(n_readouts):
        network.cell([n_cells+i]).branch(0).comp(0).record(verbose=False)

    network.delete_trainables()
    if freeze_outs:
        rec_cxns = np.nonzero(conn_matrix[:-RNN_params["n_out"], :-RNN_params["n_out"]])
        syn_inds = get_synapse_indices(network, list(rec_cxns[0]), list(rec_cxns[1]), "IonotropicSynapse")
        network.IonotropicSynapse(syn_inds).make_trainable(train_params["trainables"][0])
    else:
        network.IonotropicSynapse("all").make_trainable(train_params["trainables"][0])

    params = network.get_parameters()

    # TODO: switch to using config file
    lowers = {"IonotropicSynapse_gS": 1e-10}
    uppers = {"IonotropicSynapse_gS": np.max(abs(conn_matrix))*3}
    tf = ParamTransform(lowers, uppers)
    opt_params = tf.inverse(params)

    levels = train_params["checkpoint_levels"]

    def simulate(opt_params, stim):
        params = tf.forward(opt_params)

        tau = task_params["dt"] * 10  # another scaling of the inputs
        data_stimuli = None
        for i, w in zip(range(n_cells), init_input_weights):
            data_stimuli = network.cell(i).branch(2).comp(0).data_stimulate(
                    stim * w / tau, data_stimuli=data_stimuli
                    )
            
        num_timesteps = stim.shape[0]
        checkpoints = [int(np.ceil(num_timesteps ** (1/levels))) for _ in range(levels)]

        v = jx.integrate(
            network,
            delta_t=task_params["dt"],
            params=params,
            data_stimuli=data_stimuli,
            solver="bwd_euler",
            checkpoint_lengths=checkpoints
        )
        return v

    def predict(opt_params, stim):
        v = simulate(opt_params, stim)
        return v[-n_readouts:, 1:]

    def CE_loss(opt_params, stim, label, mask):
        pred = predict(opt_params, stim)
        # Average activity of both of the readouts in the response period
        response = jnp.mean(mask*pred, axis=1)
        # Softmax the response activations
        response = softmax(response)
        # TODO: log accuracy here?
        # Give the cross entropy loss
        return -jnp.sum(label * jnp.log(response))

    vmapped_loss_fn = vmap(CE_loss, in_axes=(None, 0, 0, 0))

    def batched_loss_fn(opt_params, stims, labels, masks):
        all_loss_vals = vmapped_loss_fn(opt_params, stims, labels, masks)
        return jnp.mean(all_loss_vals)

    grad_fn = jit(value_and_grad(batched_loss_fn, argnums=0))

    optimizer = optax.adam(learning_rate=train_params["lr"])
    opt_state = optimizer.init(opt_params)
    batch_size = train_params["batch_size"]

    stim, target, mask = gen_stim_batch(task_params, batch_size)
    loss_val, gradient = grad_fn(opt_params, stim, target, mask)
    if jnp.isnan(loss_val):
        raise Exception("Loss is NaN")
    desired_norm = l2_norm(gradient) * 0.01
    print("grad_norm: ", desired_norm)
    beta = 0.8
    losses = []

    # Make sure that the save_dir exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # Save the initial weights
    fname = os.path.join(save_dir, f"evint_init_params_{time_stamp}.pkl")
    with open(fname, "wb") as handle:
        pickle.dump((conn_matrix, init_input_weights, RNN_params), handle)
    if sync_wandb:
        wandb.save(fname)

    _ = np.random.seed(train_params["train_seed"])

    for i in tqdm(range(train_params["n_epochs"])):
        
        stim, target, mask = gen_stim_batch(task_params, batch_size)

        l, g = grad_fn(opt_params, stim, target, mask)
        g_norm = l2_norm(g)
        g = tree_map(lambda x: x / g_norm**beta * desired_norm, g) 

        updates, opt_state = optimizer.update(g, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)

        losses.append(l)

        if i % 10 == 0:
            if jnp.isnan(l):
                raise Exception("Loss is NaN")
            print(f"it {i}, loss {l}")
            if sync_wandb:
                wandb.log({"loss": l})

    # Reconstruct the final connectivity matrix
    params = tf.forward(opt_params)
    final_gS = params[0]["IonotropicSynapse_gS"]
    negative_inds = np.arange(RNN_params["n_inh"]).tolist()
    n_conn = len(network.edges["pre_cell_index"])
    final_conn_matrix = np.zeros((n_cells + n_readouts, n_cells + n_readouts))
    for i in range(n_conn):
        pre = int(network.edges.iloc[i]["pre_cell_index"])
        post = int(network.edges.iloc[i]["post_cell_index"])
        w = final_gS[i]
        if pre in negative_inds:
            assert network.edges.iloc[i, network.edges.columns.get_loc("IonotropicSynapse_e_syn")] == -75
            final_conn_matrix[pre, post] = -1 * w
        else:
            assert network.edges.iloc[i, network.edges.columns.get_loc("IonotropicSynapse_e_syn")] == 0
            final_conn_matrix[pre, post] = w
    
    fname = os.path.join(save_dir, f"evint_final_conn_{time_stamp}.pkl")
    with open(fname, "wb") as handle:
        pickle.dump(final_conn_matrix, handle)
    if sync_wandb:
        wandb.save(fname)
    
    fname = os.path.join(save_dir, f"evint_params_{time_stamp}.pkl")
    with open(fname, "wb") as handle:
        pickle.dump((params, task_params, RNN_params, train_params), handle)
    if sync_wandb:
        wandb.save(fname)

    fname = os.path.join(save_dir, f"evint_losses_{time_stamp}.pkl")
    with open(fname, "wb") as handle:
        pickle.dump(losses, handle)
    if sync_wandb:
        wandb.save(fname)

    # Average over the last few losses for the output (maybe not necessary? wandb plot thing)
    final_loss = jnp.mean(jnp.array(losses[-5:]))
    return final_loss

def main():

    time_stamp = time_module.strftime("%m%d-%H%M%S")
    print(time_stamp)

    save_dir = f"/gpfs01/berens/user/kkadhim/Jaxley/rnn_sweeps/{time_stamp}"
    os.mkdir(save_dir)

    run = wandb.init(
        project="jaxley",
        group="500ms-round2",
        name=time_stamp,
        dir=save_dir
    )

    final_loss = train(wandb.config, save_dir, time_stamp, sync_wandb=True)
    wandb.log({"final_loss": final_loss})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_id", type=str, default=None
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    if args.sweep_id is not None:
        wandb.agent(
            sweep_id=args.sweep_id,
            function=main,
            project="jaxley",
            count=1
        )
        wandb.finish()
    else:
        save_dir = "/gpfs01/berens/user/kkadhim/Jaxley/jaxley_experiments/nex/memory/trained_params"
        with open(os.path.join(save_dir, "evint_0702-100834_config.yaml")) as f:
            config = yaml.safe_load(f)
        
        save_dir = "/gpfs01/berens/user/kkadhim/Jaxley/jaxley_experiments/nex/memory/trained_params/EI_frozen_io"

        # Make the config usable by the train function
        config.pop("wandb_version")
        config.pop("_wandb")
        for key in config:
            config[key] = config[key]["value"]

        time_stamp = time_module.strftime("%m%d-%H%M%S")

        profile = False
        if profile:
            pr = cProfile.Profile()
            pr.enable()
            train(config, save_dir, time_stamp, sync_wandb=False)
            pr.disable()
            result = io.StringIO()
            pstats.Stats(pr, stream=result).print_stats()
            result = result.getvalue()
            result = 'ncalls' + result.split('ncalls')[-1]
            result = '\n'.join([','.join(line.rstrip().split(None,5)) for line in result.split('\n')])
            
            with open('train_profile.csv', 'w+') as f:
                f.write(result)
        else:
            train(config, save_dir, time_stamp, sync_wandb=False)
