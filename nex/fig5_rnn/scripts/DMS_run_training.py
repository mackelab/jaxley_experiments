import numpy as np
from nex.fig5_rnn.scripts.DMS_utils import init_and_train
seed = 3
g_scaling = 100_000 / 2 / np.pi / 10.0 / 1.0
dt = .025
sync_wandb=True
save_dir = "/mnt/qb/work/macke/mpals85/dms_mods/"
#save_dir = "/home/matthijs/jaxley_experiments/nex/memory/models/"

# Initial RNN params
RNN_params = {
    "RNN_seed":seed,
    "seed":seed,
    "n_rec":50,
    "n_inh":10,
    "n_out":2,
    "init_gain": 5/g_scaling,
    "out_scale":.2,
    "inp_scale":1,
    "in_conn_prob":.1,
    "out_conn_prob":1,
    "rec_conn_prob":.05,
    "k_minus":1,
    "out_k_minus":.1,
}

#constraints
lowers = {"input_weights": 0, "IonotropicSynapse_gS": 0, "IonotropicSynapse_k_minus": 1/20, "v": -120.0}
uppers= {"input_weights": 4, "IonotropicSynapse_gS": None, "IonotropicSynapse_k_minus": 2.0, "v":0}

training_params={
    "lr":0.001,
    "lr_end":0.0001,
    "delay_step":4000, 
    "decaying_lr":True,
    "grad_clipping":False,
    "lowers":lowers,
    "uppers":uppers,
    "max_epochs":1250,
    "loss_threshold":.65,
    "acc_threshold":.95,
    "train_v":False,
    "train_k_minus":True,
    'checkpoint_levels':2,
    "reinit_opt":False
}

#  task params (in ms)
stim_onset = [20]
stim_len = 50
delay = [50,150]
response_onset = 0
response = 50

task_params={
    "stim_onset":[int(s_o/dt) for s_o in stim_onset],
    "stim_len":int(stim_len/dt),
    "delay":[int(d/dt) for d in delay],
    "response_onset":int(response_onset/dt),
    "response":int(response/dt),
    "stim_amp":1,
    "stim_noise_sd":0.01,
    "batch_size":64

}

# run model
init_and_train(RNN_params,training_params,task_params,sync_wandb,save_dir=save_dir,device='gpu')

