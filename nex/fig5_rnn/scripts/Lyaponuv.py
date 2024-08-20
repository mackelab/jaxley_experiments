
import numpy as np
import jax.numpy as jnp
import jaxley as jx
from jax import jit, jvp


def make_trainable(network, state_keys, verbose=True,excludes = ['CaCon_e',"eCa"]):
    """Make all state keys trainable and record them"""
    network.delete_trainables()
    network.delete_recordings()
    for key in state_keys:
        if verbose:
            print(key)
        # channels
        for i in range(3):
            # check if key exsits and compartment uses this key
            for c in range(len(network.cell[i].show())):
                if key in network.cell[i].view.iloc[c] and ~np.isnan(network.cell[i].view.iloc[c][key]) and key not in excludes:
                    network.cell(c).branch(i).make_trainable(key, verbose=verbose)
                    network.cell(c).branch(i).record(key, verbose=verbose)
        # synapses
        if key in network.IonotropicSynapse("all").show():
            network.IonotropicSynapse("all").make_trainable(key, verbose=verbose)
            network.IonotropicSynapse("all").record(key)




def obtain_max_Lyapunov(network, dt=.025, T = 40000, transient = 4000, QR_t=10,v0s=None,n_repeats=1,verbose=False):
    """Calculate the Lyapunov exponent of the RNN
    Args:
        forward_func: function that takes the current state and returns the next state
        Jac_func: function that takes the current state and returns the Jacobian of the forward function
        n_exp: number of Lyapunov exponents to calculate
        T: number of time steps to calculate the Lyapunov exponents
        transient: number of time steps to discard before calculating the Lyapunov exponents
        QR_t: number of time steps between QR decompositions
        v0s: initial conditions to use
        n_repeats: number of different initial conditions to use
    Returns:
        L: maximum Lyapunov exponent
        Ls: list of Lyapunov exponents over time (e.g., to check convergence)
    """

    network.record(verbose=False)
    _, states = jx.integrate(network, return_states=True, t_max=0)
    make_trainable(network, states,verbose=verbose)

    param_dict = network.get_parameters()
    n_params = network.num_trainable_params
    param_keys = [list(param.keys())[0] for param in param_dict]
    n_per_key = [len(list(param.values())[0].flatten()) for param in param_dict]

    def param_dict_to_param_vec(param_dict):
        param_vec = []
        for item in param_dict:
            for key in item:
                param_vec.append(item[key].flatten())
        return jnp.concatenate(param_vec)

    def param_vec_to_param_dict(param_vec):
        state_dict = []
        start_idx = 0
        for key in param_keys:
            n = n_per_key[param_keys.index(key)]
            state_dict.append({key: param_vec[start_idx:start_idx+n]})
            start_idx += n
        return state_dict

    # define forward and Jacobian functions
    def one_step(param_vec):
        param_dict = param_vec_to_param_dict(param_vec)
        param_vecn =jx.integrate(network, params=param_dict, t_max=0, delta_t=dt, solver="bwd_euler")[:,-1]
        assert(len(param_vecn)==len(param_vec))
        return param_vecn
    
    forward_func = jit(one_step)
    jvp_func = lambda v,Q: jvp(forward_func, (v,), (Q,))
    jvp_func = jit(jvp_func)
    # initialise Lyapnov exponents
    L = 0
    Ls = []
    # loop through different initial conditions
    for i in range(n_repeats):

        if v0s is None:
            v = param_dict_to_param_vec(network.get_parameters())
        else:
            v = v0s[i]

        Q = np.random.randn(n_params)
        Q/=np.linalg.norm(Q)

        # discard some transient dynamics 
        # so x reaches the attractor state
        for t in range(transient):
            v = forward_func(v)
        if verbose:
            print("discarding initial transients")
        
        # discard some more transient dynamics 
        # let Q converges to the correct eigenvectors of the Oseledets matrix
        for t in range(transient):
            v,Q = jvp_func(v, Q)
            if t%QR_t == 0:
                R = np.linalg.norm(Q)
                Q = Q/R
        if verbose:
            print("computed initial Q")

        # calculate Lyapunov exponents     
        for t in range(T):
            v,Q = jvp_func(v, Q)
            if t%QR_t == 0:
                R = np.linalg.norm(Q)
                Q = Q/R
                L += np.log(R)
                Ls.append(L/(t*dt+1))
        if verbose:
            print("done, Lyapunov exponent: ", L/(T*dt*n_repeats))
    L/=(T*dt*n_repeats)
    return L, Ls
