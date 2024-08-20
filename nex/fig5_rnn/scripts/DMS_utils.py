import jaxley as jx
import numpy as np
from nex.fig5_rnn.scripts.network import initialize_RNN
import jax.numpy as jnp
import jax
from jax import jit, vmap, value_and_grad
from jax.tree_util import tree_map
import optax
import datetime
from jaxley.optimize.utils import l2_norm
import wandb
import pickle
from jax import config
from jax.lib import xla_bridge
import os
from nex.fig5_rnn.scripts.parameterisation import ParamTransform




class DMS():
    """
    Class for generating delayed match to sample task data
    """
    def __init__(self, task_params):
        self.params = task_params
    
    def gen_batch(self):
        """
        Generate a batch of trials with randomly generated delays and stimulus identities
        Returns:
            Stim: stimulus array [batch_size, duration, 2]
            target: target array [batch_size, duration, 2]
            mask: target array [batch_size, duration, 2]
        """
        n_out=2
        if len(self.params["stim_onset"])>1:
            ramp_up =(np.random.uniform(low = self.params['stim_onset'][0], high = self.params['stim_onset'][1],size=(self.params['batch_size'],)))
            ramp_up = ramp_up.astype(int)
        else:
            ramp_up = [self.params["stim_onset"][0]]*self.params['batch_size']
      
        if len(self.params["delay"])>1:
            delay =(np.random.uniform(low = self.params['delay'][0], high = self.params['delay'][1],size=(self.params['batch_size'],)))
            delay = delay.astype(int)
        else:
            delay = [self.params["delay"][0]]*self.params['batch_size']
      
        sim_time = self.params["stim_len"]*2 + self.params['delay'][-1] + self.params["stim_onset"][-1] + self.params['response_onset'] + self.params["response"]
        stim1_end = np.array(ramp_up) + self.params["stim_len"]
        delay_end = stim1_end + np.array(delay)
        stim2_end = delay_end + self.params["stim_len"]
        response_onset = stim2_end + self.params['response_onset']
        response_end = response_onset+self.params['response']

        stim = np.zeros((self.params['batch_size'], sim_time,n_out))
        target = np.zeros((self.params['batch_size'], sim_time,n_out))
        mask = np.zeros((self.params['batch_size'], sim_time,n_out))

        stim1 = np.random.randint(0,2,self.params['batch_size'])
        stim2 = np.random.randint(0,2,self.params['batch_size'])
        match = (stim1==stim2).astype(int)

        for i in range(self.params['batch_size']):
            stim[i, ramp_up[i]:stim1_end[i],stim1[i]] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
            stim[i, delay_end[i]:stim2_end[i],stim2[i]] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
            target[i,response_onset[i]:,match[i]]=1
            mask[i,response_onset[i]:response_end[i]]=1

        return stim, target, mask
    
    def gen(self):
        """
        Generate four trials, one corresponding to each possible stimulus combination
        Returns:
            Stim: stimulus array [4, duration, 2]
            target: target array [4, duration, 2]
            mask: target array [4, duration, 2]
        """
        n_trials=4
        n_out=2
        if len(self.params["stim_onset"])>1:
            ramp_up =(np.random.uniform(low = self.params['stim_onset'][0], high = self.params['stim_onset'][1],size=(n_trials,)))
            ramp_up = ramp_up.astype(int)
        else:
            ramp_up = [self.params["stim_onset"][0]]*n_trials
      
        if len(self.params["delay"])>1:
            delay =(np.random.uniform(low = self.params['delay'][0], high = self.params['delay'][1],size=(n_trials,)))
            delay = delay.astype(int)
        else:
            delay = [self.params["delay"][0]]*n_trials

      
        sim_time = self.params["stim_len"]*2 + self.params['delay'][-1] + self.params["stim_onset"][-1] + self.params['response_onset'] + self.params["response"]
        stim1_end = np.array(ramp_up) + self.params["stim_len"]
        delay_end = stim1_end + np.array(delay)
        stim2_end = delay_end + self.params["stim_len"]
        response_onset = stim2_end + self.params['response_onset']
        response_end = response_onset+self.params['response']

        stim = np.zeros((n_trials, sim_time,n_out))
        target = np.zeros((n_trials, sim_time,n_out))
        mask = np.zeros((n_trials, sim_time,1))

        stim[0, ramp_up[0]:stim1_end[0],0] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
        stim[1, ramp_up[1]:stim1_end[1],0] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
        stim[2, ramp_up[2]:stim1_end[2],1] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
        stim[3, ramp_up[3]:stim1_end[3],1] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']

        stim[0, delay_end[0]:stim2_end[0],0] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
        stim[1, delay_end[1]:stim2_end[1],1] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
        stim[2, delay_end[2]:stim2_end[2],0] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
        stim[3, delay_end[3]:stim2_end[3],1] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']

        target[0,response_onset[0]:,1]=1
        target[1,response_onset[1]:,0]=1
        target[2,response_onset[2]:,0]=1
        target[3,response_onset[3]:,1]=1

        for i in range(4):
            mask[i,response_onset[i]:response_end[i]]=1
        return stim, target, mask
   
def init_and_train(RNN_params,training_params,task_params,sync_wandb,save_dir="",device='cpu'):
    """
    Initialise and train a RNN to perform a DMS task:
    Args:
        RNN_params: dictionary with RNN parameters
        training_params: dictionary with training parameters
        task_params: dictionary with task parameters
        sync_wandb: boolean indicating whether or not to sync with wandb
        save_dir: location to save models
        device: use cpu or cuda/gpu
    
    """

    config.update("jax_enable_x64", True)
    config.update("jax_platform_name", device)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"
    os.environ["WANDB__SERVICE_WAIT"] = "500"
    
    print("using: ")
    print(xla_bridge.get_backend().platform)

    _ = np.random.seed(RNN_params['seed'])
    dt = 0.025

    # init network and task
    network, conn_matrix, input_weights=initialize_RNN(RNN_params)
    dms = DMS(task_params)

    # make network trainable
    network.IonotropicSynapse("all").make_trainable("IonotropicSynapse_gS")
    if training_params['train_k_minus']:
        network.IonotropicSynapse("all").make_trainable("IonotropicSynapse_k_minus")
    if training_params['train_v']:
        for i in range(RNN_params['n_rec']):
            network.cell(i).make_trainable("v")
    params = network.get_parameters()

    params = input_weights + params
    init_params = params.copy()
    init_params.append({'conn_matrix':conn_matrix})
    b = 1/np.mean(abs(conn_matrix.flatten()[np.nonzero(conn_matrix.flatten())]))
    tf = ParamTransform(training_params['lowers'], training_params['uppers'],bs ={'IonotropicSynapse_gS':b})
    opt_params = tf.inverse(params)
    network.delete_recordings()

    # record from readouts
    for i in range(RNN_params['n_out']):
        network.cell([RNN_params['n_rec']+i]).branch(0).comp(0).record(verbose=False)

 
    # Create simulation / training functions
    levels = training_params['checkpoint_levels']

    def simulate(opt_params, stim):
        """run simulation given stimuli"""
        params = tf.forward(opt_params)
        input_weights = params[0]["input_weights"]
        syn_weights = params[1:]

        data_stimuli = None
        for i, w in zip(range(RNN_params['n_rec']), input_weights):
            data_stimuli= network.cell(i).data_stimulate(
                jnp.inner(stim,w), data_stimuli=data_stimuli
            )
        num_timesteps = stim.shape[0]
        checkpoints = [int(np.ceil(num_timesteps ** (1/levels))) for _ in range(levels)]
        v = jx.integrate(
            network,
            delta_t=dt,
            params=syn_weights,
            data_stimuli=data_stimuli,
            solver="bwd_euler",
            checkpoint_lengths=checkpoints,
        )
        return v

    def predict(opt_params, stim):
        """extract prediction (readout units activation)"""
        v = simulate(opt_params, stim)
        return ((v[-2:])).T

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        eps=1e-8
        e_x = jnp.exp(x - jnp.max(x))
        e_x /= e_x.sum(axis=1)[:,None]
        probs = jnp.clip(e_x, eps, 1 - eps)

        return probs

    def ce(pred,label,mask):
        """ cross entropy loss"""
        #convert to probabilities with softmax:
        pred = softmax(pred)
        return -jnp.sum(mask*(label*jnp.log(pred)+(1-label)*jnp.log(1-pred)))/mask.sum()

    def accuracy(pred,label,mask):
        """calculate accuracy"""
        t_pred_1 = jnp.sum(mask[:,0]*(pred[:,1]>pred[:,0]))
        t_pred_0 = jnp.sum(mask[:,0]*(pred[:,1]<pred[:,0]))
        pred = t_pred_1>t_pred_0
        bin_label = jnp.argmax(jnp.mean(label,axis=0)) # get true label
        correct = (pred == bin_label)
        return correct.astype(int)


    def loss_fn(params, stim, label, mask):
        """calculate loss"""
        pred = predict(params, stim)
        acc = accuracy(pred,label,mask)
        loss_val = ce(pred,label,mask)
        return loss_val,acc

    vmapped_loss_fn = vmap(loss_fn, in_axes=(None, 0, 0, 0))

    def batched_loss_fn(params, stims, labels, masks):
        """mean loss over batch of trials"""
        all_loss_vals,all_accs = vmapped_loss_fn(params, stims, labels, masks)
        return jnp.mean(all_loss_vals),jnp.mean(all_accs)

    grad_fn = jit(value_and_grad(batched_loss_fn, argnums=0, has_aux=True))

    # optionally use grad clipping
    if training_params['grad_clipping']=='auto':
        stim, target, mask = dms.gen_batch()
        (l,a), g = grad_fn(opt_params, stim[:,1:], target, mask)
        grad_norm = l2_norm(g)
        print("init grad_norm", grad_norm)
        beta =.8
        desired_norm = grad_norm # The norm which you got in step 1
        grad_clipping=True
    elif training_params['grad_clipping']:
        beta=.8
        desired_norm = training_params['grad_clipping']
        grad_clipping=True
    else:
        grad_clipping=False
    
    
    # initialise optimizer
    if training_params['decaying_lr']:
        def init_opt(opt_params, lr):

            # Exponential decay of the learning rate.
            scheduler = optax.exponential_decay(
                init_value=lr,
                transition_steps=1,
                decay_rate=0.995,
                end_value=training_params['lr_end'])

            # Combining gradient transforms using `optax.chain`.
            gradient_transform = optax.chain(
                optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
                optax.scale_by_adam(),  # Use the updates from adam.
                optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
                optax.scale(-1.0)
            )
            opt_state = gradient_transform.init(opt_params)
            return opt_state, gradient_transform
    else:
        def init_opt(opt_params, lr):
            optimizer = optax.adam(learning_rate=lr)
            opt_state = optimizer.init(opt_params)
            return opt_state, optimizer

    opt_state, optimizer = init_opt(opt_params, lr = training_params['lr'])

    # start training
    dms_config={**RNN_params, **task_params, **training_params}
    if sync_wandb:
        wandb.init(project='jaxley',
                    group="DMS",
            config=dms_config)

    lowest_loss = 10
    highest_acc = 0

    for i in range(training_params['max_epochs']):
        stim, target, mask = dms.gen_batch()

        (l,a), g = grad_fn(opt_params, stim[:,1:], target, mask)
    
        if i % 10 == 0:
            print(f"it {i}, loss {l}, acc {a}")
            if sync_wandb:
                wandb.log({"loss":l,
                        "acc":a,
                        "delay":dms.params['delay'][1]})
        if np.isnan(l):
            print("loss is nan")
            break

        # Define Curriculum
        if l<training_params['loss_threshold'] and a>  training_params['acc_threshold'] and dms.params["delay"][1]<19000:
            print(f"it {i}, loss {l}, acc {a}")
            fname =save_dir+"DMS_"+str(dms.params["delay"][0])+"_"+str(dms.params["delay"][-1])+"_"+datetime.datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S")
            pickle.dump(tf.forward(opt_params), open(fname +"_params.p", "wb" ))
            pickle.dump(init_params, open(fname +"_initparams.p", "wb" ))
            pickle.dump(dms_config, open(fname +"_config.p", "wb" ))
            print("saved: " + fname +"_params.p")
            if sync_wandb:
                wandb.save(fname +"_params.p")
                wandb.save(fname +"_initparams.p")
                wandb.save(fname +"_config.p")
            if training_params['reinit_opt']:
                opt_state, optimizer = init_opt(opt_params, lr = training_params['lr'])

            dms.params["delay"][0]+=training_params['delay_step']
            dms.params["delay"][1]+=training_params['delay_step']
            

            print("increasing delay to :" + str(dms.params["delay"][1]))
        else:
            #store best performing model
            if dms.params["delay"][1]>19000 and l<training_params['loss_threshold'] and a> training_params['acc_threshold']:
                if l<=lowest_loss and a>=highest_acc:
                    print(f"it {i}, loss {l}, acc {a}")
                    fname =save_dir+"DMS_"+str(dms.params["delay"][0])+"_"+str(dms.params["delay"][-1])+"_"+datetime.datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S")
                    pickle.dump(tf.forward(opt_params), open(fname +"_params.p", "wb" ))
                    pickle.dump(init_params, open(fname +"_initparams.p", "wb" ))
                    pickle.dump(dms_config, open(fname +"_config.p", "wb" ))
                    print("saved: " + fname +"_params.p")
                    if sync_wandb:
                        wandb.save(fname +"_params.p")
                        wandb.save(fname +"_initparams.p")
                        wandb.save(fname +"_config.p")
                    lowest_loss = l
                    highest_acc = a
            
            #apply updates
            if grad_clipping:
                grad_norm = l2_norm(g)
                g= tree_map(lambda x: x / grad_norm**beta * desired_norm, g) 
            updates, opt_state = optimizer.update(g, opt_state)
            opt_params = optax.apply_updates(opt_params, updates)


                
    fname =save_dir+"DMS_"+str(dms.params["delay"][0])+"_"+str(dms.params["delay"][-1])+"_"+datetime.datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S")
    pickle.dump(tf.forward(opt_params), open(fname +"_params.p", "wb" ))
    pickle.dump(init_params, open(fname +"_initparams.p", "wb" ))
    pickle.dump(dms_config, open(fname +"_config.p", "wb" ))
    print("saved: " + fname +"_params.p")
    
    if sync_wandb:
        # store to wandb
        if sync_wandb:
            wandb.save(fname +"_params.p")
            wandb.save(fname +"_initparams.p")
            wandb.save(fname +"_config.p")
        wandb.finish()
