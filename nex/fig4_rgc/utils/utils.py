import numpy as np

import jax.numpy as jnp
import jaxley as jx
from jaxley_mech.channels.fm97 import Na, K, Leak, KA, KCa, Ca, CaNernstReversal, CaPump


def build_cell(cell_id, nseg, soma_radius, path_prefix="."):
    cell = jx.read_swc(f"{path_prefix}/morphologies/{cell_id}.swc", nseg=nseg, max_branch_len=300.0, min_radius=1.0, assign_groups=True)
    cell.set("axial_resistivity", 5_000.0)
    
    cell.insert(Na())
    cell.insert(K())
    cell.insert(Leak())
    cell.insert(KA())
    cell.insert(CaNernstReversal())
    cell.insert(CaPump())
    cell.insert(Ca())
    cell.insert(KCa())
    
    cell.soma.set("Na_gNa", 0.15)
    cell.basal.set("Na_gNa", 0.05)
    cell.set("CaPump_taur", 100.0)

    # Assume radius of soma is 10.0. Therefore, Area=4*pi*r*r, therefore length of entire soma is l=2*r.
    # Thus l of one compartment is l=2*r/nseg
    assumed_radius = soma_radius  # um
    cell.soma.set("radius", assumed_radius)
    cell.soma.set("length", 2 * assumed_radius / nseg)
    
    # Default value for radius of basal dendrites, see Fig7b of Ran et al. This value will also be learned. 
    cell.basal.set("radius", 0.2)  # um

    cell.set("v", -69.0)
    cell.init_states()

    return cell


def build_kernel(time_vec, dt):
    rise_half_time = 5  # ms
    decay_half_time = 100  # ms
    
    factor_rise = 5
    factor_decay = 3
    
    kernel_dur = factor_rise * rise_half_time + factor_decay * decay_half_time
    kernel_time_vec = np.arange(0, kernel_dur, dt)
    kernel_rise_time = np.arange(0, factor_rise * rise_half_time, dt)
    kernel_decay_time = np.arange(0, factor_decay * decay_half_time, dt)
    
    kernel_rise = 1 - np.exp(np.log(0.5) / rise_half_time * kernel_rise_time)
    
    max_val = np.max(kernel_rise)
    kernel_decay = max_val * np.exp(np.log(0.5) / decay_half_time * kernel_decay_time)
    
    kernel = np.concatenate([kernel_rise, kernel_decay])
    kernel = kernel[:len(time_vec)]
    kernel = jnp.asarray(kernel)
    return kernel
