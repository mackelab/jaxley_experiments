import collections
from typing import Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import jaxley as jx


def update_nested_dict(d, u):
    """Update nested dictionary.

    Args:
        d: Dictionary.
        u: Dictionary to update d.

    Return:
        Merged/updated dictionaries.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_synapse_indices(
    network: jx.Network,
    pre_cells: Union[int, List],
    post_cells: Union[int, List],
    synapse_type: str,
) -> List[int]:
    """Get the synapse indices in the SynapseView between the given pre and post synaptic cells.

    These synapses can then be used to index into the SynapseView of the respective
    synapse type and set parameters.
    """
    # Have to convert to lists to make the dataframe for connection identification
    pre_cells = [pre_cells] if isinstance(pre_cells, int) else pre_cells
    post_cells = [post_cells] if isinstance(post_cells, int) else post_cells
    assert len(pre_cells) == len(
        post_cells
    ), "Number of presynaptic and postsynaptic cells must be equal."
    edges_of_synapse_type = network.edges[network.edges["type"] == synapse_type]
    edge_properties_df = pd.DataFrame(
        {"pre_cell_index": pre_cells, "post_cell_index": post_cells}
    )
    # Exists will be "both" here for all edges between pre and post synaptic cell(s)
    comparison_df = pd.merge(
        edges_of_synapse_type,
        edge_properties_df,
        on=["pre_cell_index", "post_cell_index"],
        how="left",
        indicator="exists",
    )
    synapse_indices = comparison_df[
        comparison_df["exists"] == "both"
    ].index.to_numpy()
    if synapse_indices.size == 0:
        raise ValueError("No synapses found with the given edge properties.")
    return list(synapse_indices)

def update_group(ctype: str, network: jx.Network) -> pd.DataFrame:
    """Update the group view with the latest network view. 
    TODO: improve
    """
    cell_groups = {
        "Rod": network.Rod, 
        "Cone": network.Cone, 
        "HC": network.HC, 
        "BC": network.BC, 
        "AC": network.AC,
        "RGC": network.RGC
    }
    return cell_groups[ctype]
