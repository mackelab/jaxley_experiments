from functools import partial
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import jax.numpy as jnp
import jax

import pickle

from nex.rgc.utils.rgc_utils import find_spikes

resting_potential = 50
polarization = 72.0

def plot_loss_fn_summary(loss_fn, t,x_o,x_pred):

    losses = jax.vmap(loss_fn)(x_pred)
    loss_grad = jax.vmap(jax.grad(loss_fn))(x_pred[::1000])
    loss_grad_norm = jnp.linalg.norm(loss_grad, axis=-1)
    cmap = plt.get_cmap('viridis')

    fig, ax = plt.subplots(1,3, figsize=(10,2), width_ratios=[1.,0.5,0.5])
    idx = jnp.argsort(losses)
    best_losses = losses[idx[:100]]
    best_losses = jnp.abs(-best_losses) / jnp.abs(-best_losses).max()
    best_pred = x_pred[idx[:100]]
    # Plot each line with a color from the viridis colormap
    for i in range(0,100,10):
        ax[0].plot(t, best_pred[i], color=cmap(best_losses[i]), alpha=max(float(best_losses[i]),0.05))
    ax[0].plot(t,x_o,color="black", lw=2)
    ax[0].set_title("best predictives")
    l = jnp.min(losses)
    h = jnp.quantile(losses, 0.95)
    ax[1].hist(losses.flatten(), bins=100, range=(l,h))
    l = jnp.min(loss_grad_norm)
    h = jnp.quantile(loss_grad_norm, 0.95)
    ax[2].hist(loss_grad_norm.flatten(), bins=100, range=(l,h))
    ax[1].set_title("losses")
    ax[2].set_title("grad norm")

    fig.tight_layout()
    return fig

def resting_window(t):
    return [np.logical_and(t > 0, t < resting_potential)]

def polarization_window(t):
    return [np.logical_and(t > resting_potential, t < polarization)]

def spikes_window(t):
    return [np.array(t > polarization)]

def spike_windows(v, t,size=12.):
    spike_times = np.array(t[find_spikes(v)])
    spike_lower = spike_times - size/2
    spike_upper = spike_times + size/2
    
    windows = []
    for l, u in zip(spike_lower, spike_upper):
        window = np.logical_and(t > l, t < u)
        windows.append(window)
    
    return windows

def build_summary_fn(v,t, rest_summary_ord=2, pol_summary_ord=1, spikes_summary_ord=2, individual_spike_summary_ord=[1], spike_size=12.):
    windows = []
    orders = []
    if rest_summary_ord > 0:
        windows += resting_window(t)
        orders += [rest_summary_ord]
    if pol_summary_ord > 0:
        windows += polarization_window(t)
        orders += [pol_summary_ord]
    if spikes_summary_ord > 0:
        windows += spikes_window(t)
        orders += [spikes_summary_ord]
    if isinstance(individual_spike_summary_ord, int) and individual_spike_summary_ord > 0:
        s_windows = spike_windows(v,t, spike_size)
        windows += s_windows
        orders += [individual_spike_summary_ord]*len(s_windows)
    if isinstance(individual_spike_summary_ord, list):
        s_windows = spike_windows(v,t, spike_size)
        windows += s_windows[:len(individual_spike_summary_ord)]
        orders += individual_spike_summary_ord
    
    @jax.jit
    def summary_fn(x):
        stats = []
        for w, o in zip(windows, orders):
            stats.append(statistics(x[w], order=o))
        return jnp.hstack(stats)
    
    return summary_fn

def statistics(x, order=2):
    stats = []
    if order == 1:
        mean = jnp.mean(x)
        stats.append(mean)
    elif order == 2:
        mean = jnp.mean(x)
        var = jnp.var(x)
        stats.extend([mean, var])
    elif order == 3:
        mean = jnp.mean(x)
        var = jnp.var(x)
        skewness = jnp.mean(((x - mean)/jnp.sqrt(var)) ** 3)
        stats.extend([mean, var, skewness])
    elif order == 4:
        mean = jnp.mean(x)
        var = jnp.var(x)
        std = jnp.sqrt(var)
        skewness = jnp.mean(((x - mean)/std) ** 3)
        kurtosis = jnp.mean(((x - mean)/std) ** 4) - 3
        stats.append([mean, var, skewness, kurtosis])
 
    return jnp.squeeze(jnp.array(stats))



# Spike probability profile

def soft_threshold(x, eta=1, threshold=-40):
    return jax.nn.sigmoid(eta * (x - threshold))

def weight_fn(x, p=0.8, q = 0.8):
    g = jnp.where(x > 0, q**x, p**(-x))
    return g

def spike_prob(x, eta=1, threshold=-40, p=0.975, q=0.975, k=1000):
    window = weight_fn(jnp.arange(-k,k), p, q)
    window = window / jnp.sum(window)
    f = jnp.convolve(soft_threshold(x, eta, threshold) , window, mode = "same")
    return f

def diff_spike_count(x, t,eta=1, threshold=-40, p=0.975, q=0.975, k=1000):
    return jnp.trapezoid(spike_prob(x, eta, threshold, p, q, k),t)

def normalize(p, t):
    p = p / jnp.trapezoid(p,t)
    return p

def spike_prob_kl(p1, p2):
    kl = jnp.sum(p1 * jnp.log(p1/(p2 + 1e-10)))
    return kl


# Dynamic time warping



def kth_cost_diag_impl(x, y, k, cost_fn=lambda x, y: (x - y) ** 2, penalty_fn=None):
    n, m = len(x), len(y)
    start_row, start_col = (-k,0)

    # Create arrays of row and column indices
    rows = jnp.arange(n) 
    cols = jnp.arange(m)
    rows = rows + start_row
    cols = cols + start_col
    rows = jnp.where(rows < 0, n , rows)
    cols = jnp.where(cols < 0, m , cols)

    # There seems to be jax.grad bug with out of bounds indices or inf values.
    # So we define our own backpropagation rule.
    x_k = jnp.take(x, rows, fill_value=jnp.inf)
    y_k = jnp.take(y, cols, fill_value=jnp.inf)
    c_k = jax.vmap(cost_fn)(x_k, y_k)

    if penalty_fn is not None:
        penalty_k = jax.vmap(penalty_fn)(rows, cols)
        c_k = c_k + penalty_k

    return c_k, x, y


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
@partial(jax.jit, static_argnames=("cost_fn", "penalty_fn"))
def kth_cost_diag(x, y, k, cost_fn=lambda x, y: (x - y) ** 2, penalty_fn=None):
    val, _, _ = kth_cost_diag_impl(x, y, k, cost_fn, penalty_fn)
    return val

    
def kth_cost_diag_fwd(x, y, k, cost_fn, penalty_fn):
    val, x, y = kth_cost_diag_impl(x, y, k, cost_fn, penalty_fn)
    return val, (x,y,k)

def kth_cost_diag_bwd(cost_fn, penalty_fn, res, g):
    x,y,k = res
    
    rows = jnp.arange(x.shape[0]) 
    cols = jnp.arange(y.shape[0])
    rows = rows - k
    cols = cols
    
    grad_x = jax.vmap(jax.grad(cost_fn, argnums=0))(x[rows], y[cols])
    grad_y = jax.vmap(jax.grad(cost_fn, argnums=1))(x[rows], y[cols])
    
    # Initialize zero gradients for x and y
    zero_grad_x = jnp.zeros_like(x)
    zero_grad_y = jnp.zeros_like(y)

    # Scatter updates only to the indices used in the forward pass
    grad_x_full = zero_grad_x.at[rows].add(grad_x * g)
    grad_y_full = zero_grad_y.at[cols].add(grad_y * g)

    return grad_x_full, grad_y_full, None


kth_cost_diag.defvjp(kth_cost_diag_fwd, kth_cost_diag_bwd)



def pad_inf(inp, before, after):
    return jnp.pad(inp, (before, after), constant_values=jnp.inf)



@partial(jax.jit, static_argnames=("cost_fn", "penalty_fn"))
def dtw_distance(x, y, cost_fn=lambda x, y: (x - y) ** 2, penalty_fn=None):
    # Highly optimized linear memory and parallel time algorithm.
    
    H, W = x.shape[0], y.shape[0]
    x = x[::-1]  # Get antidiagonals...

    two_ago = kth_cost_diag(x, y, -H + 1, cost_fn=cost_fn, penalty_fn=penalty_fn)
    one_ago = (
        kth_cost_diag(x, y, -H + 2, cost_fn=cost_fn, penalty_fn=penalty_fn) + two_ago[0]
    )
    
    init = (
            pad_inf(two_ago, 1, 0),
            pad_inf(one_ago, 1, 0)
    )

    def scan_step(carry, k):
        two_ago, one_ago = carry
        current_antidiagonal = kth_cost_diag(
            x, y, k, cost_fn=cost_fn, penalty_fn=penalty_fn
        )

        diagonal = two_ago[:-1]
        right    = one_ago[:-1]
        down     = one_ago[1:]
        best     = jnp.min(jnp.stack([diagonal, right, down], axis=-1), axis=-1)

        next_row = best + current_antidiagonal
        next_row = pad_inf(next_row, 1, 0)

        return (one_ago, next_row), None

    ks = jnp.arange(-H + 3,H)

    carry, ys = jax.lax.scan(scan_step, init, ks, unroll=8)
    return carry[1][-1]


def softmin_impl(x, gamma=1.):
    return -gamma * jax.scipy.special.logsumexp(x / -gamma, axis=-1)

    
softmin = jax.custom_vjp(softmin_impl, nondiff_argnums=(1,))

def softmin_fwd(array, gamma):
    return softmin(array, gamma), (array / -gamma, )

def softmin_bwd(_,res, g):
    scaled_array, = res
    grad = jnp.where(jnp.isinf(scaled_array),
        jnp.zeros(scaled_array.shape),
        jax.nn.softmax(scaled_array) * jnp.expand_dims(g, -1)
    )
    return grad,

softmin.defvjp(softmin_fwd, softmin_bwd)


@partial(jax.jit, static_argnums=(2, 3, 4))
def soft_dtw_distance(
    x, y, gamma=1.0, cost_fn=lambda x, y: (x - y) ** 2, penalty_fn=None
):
    
    H, W = x.shape[0], y.shape[0]

    x = x[::-1]
    two_ago = kth_cost_diag(x, y, -H + 1, cost_fn=cost_fn, penalty_fn=penalty_fn)
    one_ago = (
        kth_cost_diag(x, y, -H + 2, cost_fn=cost_fn, penalty_fn=penalty_fn) + two_ago[0]
    )
    
    
    init = (
            pad_inf(two_ago, 1, 0),
            pad_inf(one_ago, 1, 0)
    )

    def scan_step(carry, k):
        two_ago, one_ago = carry
        current_antidiagonal = kth_cost_diag(
            x, y, k, cost_fn=cost_fn, penalty_fn=penalty_fn
        )

        diagonal = two_ago[:-1]
        right    = one_ago[:-1]
        down     = one_ago[1:]

        best     = softmin(jnp.stack([diagonal, right, down], axis=-1),gamma=gamma)

        next_row = best + current_antidiagonal
        next_row = pad_inf(next_row, 1, 0)

        return (one_ago, next_row), None

    ks = jnp.arange(-H + 3,H)

    carry, _ = jax.lax.scan(scan_step, init, ks, unroll=4)
    return carry[1][-1]




# from functools import partial


# def pad_inf(inp, before, after):
#     return jnp.pad(inp, (before, after), constant_values=jnp.inf)

# @partial(jax.jit, static_argnums=(2,3))
# def _dtw_distance(x,y, gamma=1.,cost_fn= lambda x,y: (x-y)**2):
    
#     D = jax.vmap(jax.vmap(cost_fn, in_axes=(0, None)), in_axes=(None, 0))(x, y)
    
#     H, W = x.shape[0], y.shape[0]

#     rows = []
#     for row in range(D.shape[0]):
#         rows.append( pad_inf(D[row], row, D.shape[0]-row-1) )
#     model_matrix = jnp.stack(rows, axis=1)
#     # ks = jnp.arange(-H + 3,H)
#     # model_matrix = jax.vmap(lambda k: kth_cost_diag(x[::-1],y, k, cost_fn=cost_fn))(ks)
    
#     init = (
#             pad_inf(model_matrix[0], 1, 0),
#             pad_inf(model_matrix[1] + model_matrix[0, 0], 1, 0)
#     )

#     def scan_step(carry, current_antidiagonal):
#         two_ago, one_ago = carry

#         diagonal = two_ago[:-1]
#         right    = one_ago[:-1]
#         down     = one_ago[1:]
#         best     = softmin(jnp.stack([diagonal, right, down], axis=-1), gamma=gamma)

#         next_row = best + current_antidiagonal
#         next_row = pad_inf(next_row, 1, 0)

#         return (one_ago, next_row), next_row

#     # Manual unrolling:
#     # carry = init
#     # for i, row in enumerate(model_matrix[2:]):
#     #     carry, y = scan_step(carry, row)

#     carry, ys = jax.lax.scan(scan_step, init, model_matrix[2:], unroll=4)
#     return carry[1][-1]


def window_reduce(x, reduce_fn, window_size=100, stride=50):
    ids = jnp.arange(window_size,x.shape[0], stride)
    
    def scan_fn(x, i):
        start = i - window_size
        x_val = jax.lax.dynamic_slice(x, (start,), (window_size,))
        val = reduce_fn(x_val)
        return x,val
    
    return jax.lax.scan(scan_fn, x, ids)[-1]