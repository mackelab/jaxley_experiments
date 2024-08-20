import matplotlib.pyplot as plt
import numpy as np


def rgb_to_hex(rgba_value):
    return '#%02x%02x%02x' % tuple((np.asarray(rgba_value)*255)[:-1].astype(int).tolist())

import matplotlib.colors as colors


def truncate_invert_cmap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(maxval, minval, n)))
    return new_cmap

data_cmap = plt.get_cmap("viridis")
parameter_cmap = truncate_invert_cmap(plt.get_cmap("Greens"), 0.2, 1.0)
jaxley_cmap = plt.get_cmap("viridis")

cols = {
    "jaxley_shades": [
        "#d0d1e6",
        "#a6bddb",
        "#74a9cf",
        "#3690c0",
        "#0570b0",
        "#045a8d",
        "#023858",
    ],
    "jaxley": "#0570b0",
    "jaxley_cmap": jaxley_cmap,
    "jaxley_classes": [
        rgb_to_hex(plt.get_cmap(jaxley_cmap)(0.2)),
        rgb_to_hex(plt.get_cmap(jaxley_cmap)(0.95)),
    ],
    "genetic_alg": "k",
    "NEURON": "k",
    "data": "k",
    "ground_truth": "k",
    "data_classes": [
        rgb_to_hex(plt.get_cmap(data_cmap)(0.2)),
        rgb_to_hex(plt.get_cmap(data_cmap)(0.95)),
    ],
    "data_cmap": data_cmap,
    "soma": "k",
    "axon": "k",
    "apical": "k",
    "basal": "k",
    "parameter_cmap": parameter_cmap,
}
