import numpy as np
import jax.numpy as jnp


def find_spikes(v, t=None, T_stim=None, thr=-20.0, T_lockout=2.0):
    diff = jnp.diff(v)
    where_diff0 = jnp.where(jnp.logical_and(diff[:-1] > 1e-5, diff[1:] < -1e-5), True, False)
    where_spike = jnp.logical_and(where_diff0, v[1:-1] > thr)
    where_spike = jnp.pad(where_spike, (1,1), constant_values=False)

    diff2 = jnp.diff(v, 2)
    where_peak = jnp.where(diff2 < -1e-3, True, False)
    where_peak = jnp.pad(where_peak, (2,0), constant_values=False)
    where_spike = jnp.logical_and(where_spike, where_peak)

    if t is not None:
        t = jnp.array(t)
        isi = jnp.diff(t[where_spike])
        spike_idxs = jnp.where(where_spike)[0]
        is_double = spike_idxs[1:][isi < T_lockout]
        where_spike = where_spike.at[is_double].set(False)
    
    if T_stim is not None:
        during_stim = jnp.logical_and(T_stim[0] < t, t < T_stim[1])
        where_spike = jnp.logical_and(where_spike, during_stim)
    return where_spike


def spike_count(trace_spikes):
    return np.sum(trace_spikes)


def first_spike_time(trace_spikes):
    spike_inds = np.where(trace_spikes)[0]
    if len(spike_inds) > 0:
        return spike_inds[0] * 0.025
    else:
        return -100.0


def second_spike_time(trace_spikes):
    spike_inds = np.where(trace_spikes)[0]
    if len(spike_inds) > 1:
        return spike_inds[1] * 0.025
    else:
        return -100.0


def first_spike_amp(trace, trace_spikes):
    spike_inds = np.where(trace_spikes)[0]
    if len(spike_inds) > 0:
        return trace[spike_inds[0]]
    else:
        return -100.0


def second_spike_amp(trace, trace_spikes):
    spike_inds = np.where(trace_spikes)[0]
    if len(spike_inds) > 1:
        return trace[spike_inds[1]]
    else:
        return -100.0
