import jaxley as jx
from jaxley.channels import Leak, HH
from jaxley.connect import connect
from jaxley.synapses import IonotropicSynapse
from jaxley_mech.channels.l5pc import (
    NaTs2T,
    CaHVA,
    CaLVA,
    CaPump,
    CaNernstReversal,
    SKv3_1,
    SKE2,
    M,
    H,
)
from jaxley_mech.channels.fm97 import Leak
import numpy as np
from nex.fig5_rnn.scripts.compute_EI import init_inh_ex_gS
import jax.numpy as jnp

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"
os.environ["WANDB__SERVICE_WAIT"] = "300"


def initialize_RNN(params):
    n_cells = params["n_rec"]
    n_inh = params["n_inh"]
    n_out = params["n_out"]
    init_gain = params["init_gain"]
    out_scale = params["out_scale"]
    inp_scale = params["inp_scale"]
    in_conn_prob = params["in_conn_prob"]
    out_conn_prob = params["out_conn_prob"]
    rec_conn_prob = params["rec_conn_prob"]
    k_minus = params["k_minus"]
    out_k_minus = params["out_k_minus"]

    _ = np.random.seed(params["RNN_seed"])

    # Define the cell
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=1)
    cell = jx.Cell([branch, branch, branch], parents=[-1, 0, 0])

    cell.branch(0).add_to_group("apical")
    cell.branch(1).add_to_group("soma")
    cell.branch(2).add_to_group("basal")
    cell.compute_xyz()

    # APICAL
    # Sodium
    cell.apical.insert(NaTs2T())
    cell.apical.set("NaTs2T_gNaTs2T", 0.026145)
    # Potassium
    cell.apical.insert(SKv3_1())
    cell.apical.set("SKv3_1_gSKv3_1", 0.004226)
    cell.apical.insert(M())
    cell.apical.set("M_gM", 0.000143)
    # Nonspecific cation
    cell.apical.insert(H())
    for c in [0.125, 0.375, 0.625, 0.875]:
        distance = cell.branch(0).loc(c).distance(cell.branch(0).loc(0.0))
        cond = (-0.8696 + 2.087 * np.exp(distance * 0.0031)) * 8e-5
        cell.branch(0).loc(c).set("H_gH", cond)

    cell.apical.set("capacitance", 2.0)

    # SOMA
    # Calcium
    ca_dynamics = CaNernstReversal()
    ca_dynamics.channel_constants["T"] = 307.15
    cell.soma.insert(ca_dynamics)
    cell.soma.set("CaCon_i", 5e-05)
    cell.soma.set("CaCon_e", 2.0)
    cell.soma.insert(CaPump())
    cell.soma.set("CaPump_gamma", 0.000609)
    cell.soma.set("CaPump_decay", 210.485291)
    cell.soma.insert(CaHVA())
    cell.soma.set("CaHVA_gCaHVA", 0.000994)
    cell.soma.insert(CaLVA())
    cell.soma.set("CaLVA_gCaLVA", 0.000333)
    # Potassium
    cell.soma.insert(SKv3_1())
    cell.soma.set("SKv3_1_gSKv3_1", 0.303472)
    cell.soma.insert(SKE2())
    cell.soma.set("SKE2_gSKE2", 0.008407)
    # Sodium
    cell.soma.insert(NaTs2T())
    cell.soma.set("NaTs2T_gNaTs2T", 0.983955)

    # BASAL
    # Nonspecific cation
    cell.basal.insert(H())
    cell.basal.set("H_gH", 8e-5)

    # GLOBAL
    cell.insert(Leak())
    cell.set("Leak_gLeak", 3e-05)
    cell.set("Leak_eLeak", -75.0)
    cell.set("axial_resistivity", 100.0)
    cell.set("eNa", 50.0)
    cell.set("eK", -85.0)

    readout = jx.Cell([branch], parents=[-1])
    readout.insert(Leak())
    readout.set("Leak_gLeak", 3e-2)
    readout.show()

    # Define the network
    network = jx.Network([cell for _ in range(n_cells)] + [readout]*n_out)

    # Connect the somas to the apical dendrites
    for i in range(n_cells):
        for j in range(n_cells):
            if np.random.binomial(1, rec_conn_prob):
                connect(
                    network.cell(i).branch(1).comp(0),
                    network.cell(j).branch(0).comp(0),
                    IonotropicSynapse(),
                )
        # Connect the cell to the readouts
        if n_out:
            if out_conn_prob < 1.0:
                if np.random.binomial(1, out_conn_prob):
                    connect(
                        network.cell(i).branch(1).comp(0),
                        network.cell(n_cells).branch(0).comp(0),
                        IonotropicSynapse(),
                    )
                if np.random.binomial(1, out_conn_prob):
                    connect(
                        network.cell(i).branch(1).comp(0),
                        network.cell(n_cells + 1).branch(0).comp(0),
                        IonotropicSynapse(),
                    )
            else:
                connect(
                    network.cell(i).branch(1).comp(0),
                    network.cell(n_cells).branch(0).comp(0),
                    IonotropicSynapse(),
                )
                connect(
                    network.cell(i).branch(1).comp(0),
                    network.cell(n_cells + 1).branch(0).comp(0),
                    IonotropicSynapse(),
                )

    # Define inhibitory vs excitatory connectivity
    negative_inds = np.arange(n_inh).tolist()
    conn_matrix = init_inh_ex_gS(
        network,
        negative_inds,
        init_gain,
        out_indices=[n_cells, n_cells + 1],
        out_scale=out_scale,
        return_matrix=True,
        dist="normal",
    )

    input_weights = np.zeros((n_cells,n_out))
    if n_out:
        # Initialization of input weights sparse and from a uniform distribution
        if in_conn_prob < 1.0:
            for i in range(n_cells):
                # if np.random.rand()>p_in_conn*2:
                #    input_weights[i,np.random.randint(0,n_out)]=1
                if np.random.rand() < in_conn_prob:
                    input_weights[i, 0] = abs(np.random.uniform(0, 1))
                if np.random.rand() < in_conn_prob:
                    input_weights[i, 1] = abs(np.random.uniform(0, 1))
            input_weights = [{"input_weights": jnp.asarray(input_weights * inp_scale)}]
        # Initialization of input weights not sparse and from a normal distribution
        else:
            input_weights = [
                {"input_weights": jnp.asarray(np.random.randn(n_cells) * inp_scale)}
            ]

    # Set the k_minus values
    n_conn = len(network.edges["pre_cell_index"])
    for i in range(n_conn):
        post = int(network.edges.iloc[i]["post_cell_index"])
        if post < n_cells:
            # k_mins = 1.0 still ok for in the recurrent network for now
            network.edges.iloc[
                i, network.edges.columns.get_loc("IonotropicSynapse_k_minus")
            ] = k_minus
        else:
            # Lower k_minus for the leak neuron readout
            network.edges.iloc[
                i, network.edges.columns.get_loc("IonotropicSynapse_k_minus")
            ] = out_k_minus

    # Set some parameters
    network.set("v", -65.0)
    network.init_states()

    return network, conn_matrix, input_weights
