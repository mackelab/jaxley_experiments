import jaxley as jx


transform_params = jx.ParamTransform(
    lowers={
        "axial_resistivity": 100.0,
        "radius": 0.1,
        "w_bc_to_rgc": 0.0,
    },
    uppers={
        "axial_resistivity": 10_000.0,
        "radius": 1.0,
        "w_bc_to_rgc": 0.2,
    },
)

transform_basal = jx.ParamTransform(
    lowers={
        "Na_gNa": 0.0,
        "K_gK": 0.01,
        "Leak_gLeak": 1e-5,
        "KA_gKA": 10e-3,
        "Ca_gCa": 2e-3,
        "KCa_gKCa": 0.02e-3,
    },
    uppers={
        "Na_gNa": 0.5,
        "K_gK": 0.1,
        "Leak_gLeak": 1e-3,
        "KA_gKA": 100e-3,
        "Ca_gCa": 3e-3,
        "KCa_gKCa": 0.2e-3,
    },
)

transform_somatic = jx.ParamTransform(
    lowers={
        "Na_gNa": 0.05,
        "K_gK": 0.01,
        "Leak_gLeak": 1e-5,
        "KA_gKA": 10e-3,
        "Ca_gCa": 2e-3,
        "KCa_gKCa": 0.02e-3,
    },
    uppers={
        "Na_gNa": 0.5,
        "K_gK": 0.1,
        "Leak_gLeak": 1e-3,
        "KA_gKA": 100e-3,
        "Ca_gCa": 3e-3,
        "KCa_gKCa": 0.2e-3,
    },
)