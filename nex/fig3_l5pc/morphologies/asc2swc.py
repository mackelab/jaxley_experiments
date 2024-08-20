import os
from morph_tool import convert


IN_DIR = "/Users/michaeldeistler/Documents/phd/jaxley_experiments/nex/l5pc/morphologies"
OUT_DIR = "/Users/michaeldeistler/Documents/phd/jaxley_experiments/nex/l5pc/morphologies"
fnames = ["bbp_with_axon.asc", "bbp_no_axon.asc"]
SANITIZED = True
SINGLE_POINT_SOMA = False


for fname in fnames:
    in_path = os.path.join(IN_DIR, fname)
    out_path = os.path.join(OUT_DIR, fname[:-3] + "swc")
    convert(in_path, out_path, sanitize=SANITIZED, single_point_soma=SINGLE_POINT_SOMA)
