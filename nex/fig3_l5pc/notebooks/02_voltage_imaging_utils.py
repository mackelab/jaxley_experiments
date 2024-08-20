from jax import config

config.update("jax_enable_x64", True)
#config.update("jax_platform_name", "cpu")

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import time
from itertools import chain
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad, lax
from jax.tree_util import tree_map
import jax
import jaxlib
import pandas as pd
import optax
from itertools import chain

import jaxley as jx
from jaxley.channels import Leak
from jaxley_mech.channels.l5pc import *
from jaxley.optimize.utils import l2_norm

def sample_within(n, bounds):
    return jnp.asarray(np.random.rand(n) * (bounds[1] - bounds[0]) + bounds[0])

bounds = {}
gt_apical = {}
gt_soma = {}
gt_axon = {}

bounds["apical_NaTs2T_gNaTs2T"] = [0, 0.04]
gt_apical["apical_NaTs2T_gNaTs2T"] = 0.026145

bounds["apical_SKv3_1_gSKv3_1"] = [0, 0.04]
gt_apical["apical_SKv3_1_gSKv3_1"] = 0.004226

bounds["apical_M_gM"] = [0, 0.001]
gt_apical["apical_M_gM"] = 0.000143

bounds["somatic_NaTs2T_gNaTs2T"] = [0.0, 1.0]
gt_soma["somatic_NaTs2T_gNaTs2T"] = 0.983955

bounds["somatic_SKv3_1_gSKv3_1"] = [0.25, 1]
gt_soma["somatic_SKv3_1_gSKv3_1"] = 0.303472

bounds["somatic_SKE2_gSKE2"] = [0, 0.1]
gt_soma["somatic_SKE2_gSKE2"] = 0.008407

bounds["somatic_CaPump_gamma"] = [0.0005,0.01]
gt_soma["somatic_CaPump_gamma"] = 0.000609

bounds["somatic_CaPump_decay"] = [20, 1_000]
gt_soma["somatic_CaPump_decay"] = 210.485291

bounds["somatic_CaHVA_gCaHVA"] = [0, 0.001]
gt_soma["somatic_CaHVA_gCaHVA"] = 0.000994

bounds["somatic_CaLVA_gCaLVA"] = [0, 0.01]
gt_soma["somatic_CaLVA_gCaLVA"] = 0.000333

bounds["axonal_NaTaT_gNaTaT"] = [0.0, 4.0]
gt_axon["axonal_NaTaT_gNaTaT"] = 3.137968

bounds["axonal_KPst_gKPst"] = [0.0, 1.0]
gt_axon["axonal_KPst_gKPst"] = 0.973538

bounds["axonal_KTst_gKTst"] = [0.0, 0.1]
gt_axon["axonal_KTst_gKTst"] = 0.089259

bounds["axonal_SKE2_gSKE2"] = [0.0, 0.1]
gt_axon["axonal_SKE2_gSKE2"] = 0.007104

bounds["axonal_SKv3_1_gSKv3_1"] = [0.0, 2.0]
gt_axon["axonal_SKv3_1_gSKv3_1"] = 1.021945

bounds["axonal_CaHVA_gCaHVA"] = [0, 0.001]
gt_axon["axonal_CaHVA_gCaHVA"] = 0.00099

bounds["axonal_CaLVA_gCaLVA"] = [0, 0.01]
gt_axon["axonal_CaLVA_gCaLVA"] = 0.008752

bounds["axonal_CaPump_gamma"] = [0.0005, 0.05]
gt_axon["axonal_CaPump_gamma"] = 0.00291

bounds["axonal_CaPump_decay"] = [20, 1_000]
gt_axon["axonal_CaPump_decay"] = 287.19873


res = 100
sigma = 400

evals = np.linspace(0, 1200, res)
kernel = np.zeros((res, res))
for ind_i, i in enumerate(evals):
    for ind_j, j in enumerate(evals):
        d = (i - j)**2
        kernel[ind_i, ind_j] = np.exp(-d / sigma**2)

def softplus(x):
    return np.log(1 + np.exp(x))

def inv_softplus(x: jnp.ndarray):
    """Inverse softplus."""
    return np.log(np.exp(x) - 1)

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid."""
    return 1 / (1 + jnp.exp(-1.5*x))

def expit(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse sigmoid (expit)"""
    return -jnp.log(1 / x - 1)

def gaussian_cdf(x):
    return jax.scipy.stats.norm.cdf(x)

def sample_profile():
    profile = np.random.multivariate_normal(np.zeros((res)), kernel)
    return gaussian_cdf(profile) * 0.8 + 0.1  # ground truth is bounded from [0.1, 0.9].

# For every parameter, sample the ground truth parameter function.
np.random.seed(2)
gt_profiles_apical = {}
gt_profiles_apical["apical_NaTs2T_gNaTs2T"] = sample_profile()
gt_profiles_apical["apical_SKv3_1_gSKv3_1"] = sample_profile()
gt_profiles_apical["apical_M_gM"] = sample_profile()

np.random.seed(22)
gt_profiles_axonal = {}
gt_profiles_axonal["axonal_NaTaT_gNaTaT"] = sample_profile()
gt_profiles_axonal["axonal_KPst_gKPst"] = sample_profile()
gt_profiles_axonal["axonal_KTst_gKTst"] = sample_profile()
gt_profiles_axonal["axonal_SKE2_gSKE2"] = sample_profile()
gt_profiles_axonal["axonal_SKv3_1_gSKv3_1"] = sample_profile()
gt_profiles_axonal["axonal_CaHVA_gCaHVA"] = sample_profile()
gt_profiles_axonal["axonal_CaLVA_gCaLVA"] = sample_profile()


apical_inds = []
soma_inds = []
basal_inds = []
axonal_inds = []

def build_cell():
    nseg = 4
    cell = jx.read_swc("../../paper/fig3_l5pc/morphologies/bbp_with_axon.swc", nseg=nseg, max_branch_len=300.0, assign_groups=True)

    global apical_inds, soma_inds, basal_inds, axonal_inds

    soma_inds = np.unique(cell.group_nodes["soma"].branch_index).tolist()
    apical_inds = np.unique(cell.group_nodes["apical"].branch_index).tolist()
    basal_inds = np.unique(cell.group_nodes["basal"].branch_index).tolist()
    axonal_inds = np.unique(cell.group_nodes["axon"].branch_index).tolist()

    ########## APICAL ##########
    cell.apical.set("capacitance", 2.0)
    cell.apical.insert(NaTs2T().change_name("apical_NaTs2T"))
    cell.apical.insert(SKv3_1().change_name("apical_SKv3_1"))
    cell.apical.insert(M().change_name("apical_M"))
    cell.apical.insert(H().change_name("apical_H"))
    for b in apical_inds:
        for c in [0.125, 0.375, 0.625, 0.875]:
            distance = cell.branch(b).loc(c).distance(cell.branch(0).loc(0.0))
            cond = (-0.8696 + 2.087* np.exp(distance*0.0031)) * 8e-5
            cell.branch(b).loc(c).set("apical_H_gH", cond)

    ########## SOMA ##########
    cell.soma.insert(NaTs2T().change_name("somatic_NaTs2T"))
    cell.soma.insert(SKv3_1().change_name("somatic_SKv3_1"))
    cell.soma.insert(SKE2().change_name("somatic_SKE2"))
    ca_dynamics = CaNernstReversal()
    ca_dynamics.channel_constants["T"] = 307.15
    cell.soma.insert(ca_dynamics)
    cell.soma.insert(CaPump().change_name("somatic_CaPump"))
    cell.soma.insert(CaHVA().change_name("somatic_CaHVA"))
    cell.soma.insert(CaLVA().change_name("somatic_CaLVA"))
    cell.soma.set("CaCon_i", 5e-05)
    cell.soma.set("CaCon_e", 2.0)

    ########## BASAL ##########
    cell.basal.insert(H().change_name("basal_H"))
    cell.basal.set("basal_H_gH", 8e-5)

    # ########## AXON ##########
    cell.insert(CaNernstReversal())
    cell.set("CaCon_i", 5e-05)
    cell.set("CaCon_e", 2.0)

    cell.axon.insert(NaTaT().change_name("axonal_NaTaT"))
    cell.axon.set("axonal_NaTaT_gNaTaT", 3.137968)

    cell.axon.insert(KTst().change_name("axonal_KTst"))
    cell.axon.set("axonal_KTst_gKTst", 0.089259)

    cell.axon.insert(CaPump().change_name("axonal_CaPump"))
    cell.axon.set("axonal_CaPump_gamma", 0.00291)
    cell.axon.set("axonal_CaPump_decay", 287.19873)

    cell.axon.insert(SKE2().change_name("axonal_SKE2"))
    cell.axon.set("axonal_SKE2_gSKE2", 0.007104)

    cell.axon.insert(CaHVA().change_name("axonal_CaHVA"))
    cell.axon.set("axonal_CaHVA_gCaHVA", 0.00099)

    cell.axon.insert(KPst().change_name("axonal_KPst"))
    cell.axon.set("axonal_KPst_gKPst", 0.973538)

    cell.axon.insert(SKv3_1().change_name("axonal_SKv3_1"))
    cell.axon.set("axonal_SKv3_1_gSKv3_1", 1.021945)

    cell.axon.insert(CaLVA().change_name("axonal_CaLVA"))
    cell.axon.set("axonal_CaLVA_gCaLVA", 0.008752)


    ########## WHOLE CELL  ##########
    cell.insert(Leak())
    cell.set("Leak_gLeak", 3e-05)
    cell.set("Leak_eLeak", -75.0)

    cell.set("axial_resistivity", 100.0)
    cell.set("eNa", 50.0)
    cell.set("eK", -85.0)
    cell.set("v", -65.0)


    # Apical.
    for b in apical_inds:
        distance = cell.branch(b).loc(0.0).distance(cell.branch(0).loc(0.0))
        for key, item in gt_profiles_apical.items():
            cond = np.interp(distance, evals, item)
            cell.branch(b).set(key, cond * bounds[key][1])

    for key in gt_soma.keys():
        cell.soma.set(key, gt_soma[key])

    for b in axonal_inds:
        distance = cell.branch(b).loc(0.0).distance(cell.branch(0).loc(0.0))
        for key, item in gt_profiles_axonal.items():
            cond = np.interp(distance, evals, item)
            cell.branch(b).set(key, cond * bounds[key][1])

    for key in gt_axon.keys():
        cell.axon.set(key, gt_axon[key])
        
        
    cell.delete_trainables()

    for key in gt_profiles_apical.keys():
        cell.branch(apical_inds).make_trainable(key)

    for key in gt_soma.keys():
        cell.soma.make_trainable(key)

    for key in gt_profiles_axonal.keys():
        cell.branch(axonal_inds).make_trainable(key)

    cell.axon.make_trainable("axonal_CaPump_gamma")
    cell.axon.make_trainable("axonal_CaPump_decay")

    parameters = cell.get_parameters()
    
    repeats = {
        "apical_NaTs2T_gNaTs2T": len(apical_inds),
        "apical_SKv3_1_gSKv3_1": len(apical_inds),
        "apical_M_gM": len(apical_inds), 
        "somatic_NaTs2T_gNaTs2T": 1,
        "somatic_SKv3_1_gSKv3_1": 1,
        "somatic_SKE2_gSKE2": 1,
        "somatic_CaHVA_gCaHVA": 1,
        "somatic_CaLVA_gCaLVA": 1,
        "somatic_CaPump_gamma": 1,
        "somatic_CaPump_decay": 1,
        "axonal_NaTaT_gNaTaT": len(axonal_inds),
        "axonal_KPst_gKPst": len(axonal_inds),
        "axonal_KTst_gKTst": len(axonal_inds),
        "axonal_SKE2_gSKE2": len(axonal_inds),
        "axonal_SKv3_1_gSKv3_1": len(axonal_inds),
        "axonal_CaHVA_gCaHVA": len(axonal_inds),
        "axonal_CaLVA_gCaLVA": len(axonal_inds),
        "axonal_CaPump_gamma": 1,
        "axonal_CaPump_decay": 1,
    }
    num_params = sum(list(repeats.values()))
    print(f"Total number of parameters: {num_params}")
    
    dt = 0.025
    t_max = 5.0
    time_vec = np.arange(0, t_max+2*dt, dt)

    cell.delete_stimuli()
    cell.delete_recordings()

    i_delay = 0.0
    i_dur = 3.0
    i_amp = 3.0
    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    cell.branch(soma_inds[0]).loc(0.0).stimulate(current)  # Stimulate soma
    cell[range(0,339,1),0].record()

    cell.set("v", -70.0)
    cell.init_states()
    
    return cell, num_params, parameters, bounds

def get_transforms(paramters, bounds):
    lowers = {}
    uppers = {}
    for key in bounds:
        lowers[key] = bounds[key][0]
        uppers[key] = bounds[key][1]

    transform = jx.ParamTransform(
        lowers=lowers,
        uppers=uppers,
    )
    
    _, unflatten = jax.flatten_util.ravel_pytree(paramters)
    
    return unflatten, transform

def get_simulator_and_xo(cell):
    
    @jax.jit
    def simulate(params):
        return jx.integrate(cell, params=params, checkpoint_lengths=[101,2])
    
    @jax.jit
    def sim_gt():
        return jx.integrate(cell)
    
    x_o = sim_gt()

    return simulate, x_o


def get_loss_and_regularizer(cell, x_o):
    def build_regularizer(inds):
        parents = cell.comb_parents
        apical_inds_from_zero = np.arange(len(inds))
        
        parents_of_apical = np.asarray(parents)[np.asarray(inds)]
        parent_is_also_apical = np.asarray([p in np.asarray(inds) for p in parents_of_apical])
        apical_inds_from_zero = apical_inds_from_zero[parent_is_also_apical]
        
        parents_of_apical = parents_of_apical[parent_is_also_apical]
        parents_from_zero = np.asarray([np.where(inds == p)[0][0] for p in parents_of_apical])
        
        return apical_inds_from_zero, parents_from_zero
    
    param_child_apical, param_parent_apical = build_regularizer(apical_inds)
    param_child_axonal, param_parent_axonal = build_regularizer(axonal_inds)
    
    def loss_from_v(v):
        return jnp.mean(jnp.abs(v[:, 40:200:5] - x_o[:, 40:200:5]))
    
    def regularizer(opt_params):
        reg_apical = 0.0
        reg_axonal = 0.0
        for key, i in zip(gt_profiles_apical, [0, 1, 2]):
            reg_apical += jnp.sum((opt_params[i][key][param_child_apical] - opt_params[i][key][param_parent_apical])**2)
        for key, i in zip(gt_profiles_axonal, np.arange(10, 20)):
            reg_axonal += jnp.sum((opt_params[i][key][param_child_axonal] - opt_params[i][key][param_parent_axonal])**2)
        return reg_apical + reg_axonal
    
    return loss_from_v, regularizer



