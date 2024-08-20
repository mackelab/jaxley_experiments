import numpy as np
import jax.numpy as jnp
from scipy.signal import butter, lfilter



def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def gen_stim(task_params, return_offsets=False, return_mean=False, stim_mean_dist="normal"):
    # Filter of the stimulus
    cutoff = 5 # desired cutoff frequency of the filter, mHz
    order = 6
    fs = task_params["stim_freq_max"]

    # Make the stimulus
    if stim_mean_dist == "normal":
        mean_offset = np.random.binomial(1, 0.5)
        stim_mean = np.random.normal(task_params["stim_mean"][mean_offset], task_params["stim_mean_std"])
    if stim_mean_dist == "uniform":
        stim_mean = np.random.uniform(task_params["stim_mean"][0], task_params["stim_mean"][1]) 
    
    stim_std = task_params["stim_std"]
    n = int(task_params["total_time"] / task_params["dt"])
    stim_unfiltered = np.random.normal(stim_mean, stim_std, size=n)
    stim = butter_lowpass_filter(stim_unfiltered, cutoff, fs, order)

    # Build the mask
    mask = np.zeros_like(stim)
    start_response_dt = int(task_params["response_time"] / task_params["dt"])
    end_response_dt = int((task_params["response_time"] + task_params["response_length"]) / task_params["dt"])
    mask[start_response_dt:end_response_dt] = 1

    # Calculate the target
    offsets = jnp.sum(stim * np.invert(mask.astype(bool)).astype(int))
    
    if task_params["loss"] == "MSE":
        depol_response = np.ones_like(stim) * -70
        depol_response[start_response_dt:end_response_dt] = task_params["target_pol"]
        rest_response = np.ones_like(stim) * -70
        if jnp.sign(offsets) > 0:
            target_response = np.stack([depol_response, rest_response])
        else:
            target_response = np.stack([rest_response, depol_response])
    elif task_params["loss"] == "CE":
        if jnp.sign(offsets) > 0:
            target_response = np.array([1, 0])
        else:
            target_response = np.array([0, 1])
            
    data = [stim, target_response, mask]
    if return_offsets:
        data.append(offsets)
    if return_mean:
        data.append(stim_mean)
    return data

def gen_stim_batch(task_params, batch_size):
    stims = []
    targets = []
    masks = []
    for _ in range(batch_size):
        stim, target_response, mask = gen_stim(task_params)
        stims.append(stim) 
        targets.append(target_response)
        masks.append(mask)
    return jnp.array(stims), jnp.array(targets), jnp.array(masks)


