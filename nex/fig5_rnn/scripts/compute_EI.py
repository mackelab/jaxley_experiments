import jax.numpy as jnp
import numpy as np

def init_inh_ex_gS(
    network, inh_indices, gain, out_indices = [],e_syn_inh=-75, e_syn_ex=0, out_scale=1, return_matrix=False,dist = "normal",name = "IonotropicSynapse"
):
    """
    Initialises an excitatory inhibitory network with random weights, such that EI balance is preserved
    """

    n_units = (
        max(
            np.max(network.edges["pre_cell_index"]),
            np.max(network.edges["post_cell_index"]),
        )
        + 1
    )
    n_out = len(out_indices)
    n_units_rec = n_units-n_out
    n_inh = len(inh_indices)
    rec_inds = [ind for ind in range(n_units) if ind not in out_indices]
    recurrent_post_indices = [ind for ind in network.edges["post_cell_index"] if ind not in out_indices]
    n_conn_rec = len(recurrent_post_indices)
    n_conn = len(network.edges["pre_cell_index"])

    average_in_rec = n_conn_rec / n_units_rec
    #print(n_inh,n_units_rec)
    p_inh = n_inh / n_units_rec
    n_ex = n_units_rec - n_inh
    print(average_in_rec)
    print(n_conn_rec)
    print(n_units_rec)
    p_rec = average_in_rec / n_units_rec
    print("conn probability recurrence: " + str(p_rec))
    conn_matrix = np.zeros((n_units, n_units))

    EIratio = (1 - p_inh) / (p_inh)
    print("EIratio:" + str(EIratio))

    normaliser = np.sqrt((1 / (1 - (2 * p_rec) / np.pi)) / EIratio)
    print("Normaliser: " + str(normaliser))
    # this normaliser scales the variances of the two half normal distributions to be gain**2 / N

    for i in range(n_conn):
        pre = int(network.edges.iloc[i]["pre_cell_index"])
        post = int(network.edges.iloc[i]["post_cell_index"])
        if dist=="normal":
            samp = abs(np.random.normal(0, 1, 1)[0])
        elif dist =="uniform":
            samp = abs(np.random.uniform(0, 1))
        else:
            print("dist not recognised, use normal or uniform")
        if post in out_indices:
            samp*=out_scale

        if pre in inh_indices:
            w = samp * normaliser * gain * EIratio / np.sqrt(average_in_rec)
            network.edges.iloc[
                i, network.edges.columns.get_loc(name+"_gS")
            ] = w
            network.edges.iloc[i, network.edges.columns.get_loc(name+"_e_syn")] = e_syn_inh
            conn_matrix[pre, post] = w * -1

        else:
            w = samp * normaliser * gain / np.sqrt(average_in_rec)
            network.edges.iloc[
                i, network.edges.columns.get_loc(name+"_gS")
            ] = w
            network.edges.iloc[i, network.edges.columns.get_loc(name+"_e_syn")] = e_syn_ex
            conn_matrix[pre, post] = w

    ev = np.linalg.eigvals(conn_matrix[rec_inds][:,rec_inds])
    print("Spectral radius recurrence: " + str(np.max(np.abs(ev))))
    # this should be close to gain!

    if return_matrix:
        return conn_matrix