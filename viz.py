import numpy as np
import biorbd_casadi as biorbd


def plot_com(x, nlp):
    com_func = biorbd.to_casadi_func("CoMPlot", nlp.model.CoM, nlp.states["q"].mx, expand=False)
    com_dot_func = biorbd.to_casadi_func(
        "Compute_CoM", nlp.model.CoMdot, nlp.states["q"].mx, nlp.states["qdot"].mx, expand=False
    )
    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])

    return np.concatenate((np.array(com_func(q)[1:, :]), np.array(com_dot_func(q, qdot)[1:, :])))


def add_custom_plots(ocp):
    for i, nlp in enumerate(ocp.nlp):
        ocp.add_plot(
            "CoM", lambda t, x, u, p: plot_com(x, nlp), phase=i, legend=["CoMy", "Comz", "CoM_doty", "CoM_dotz"]
        )

        # ocp.add_plot_penalty(CostType.ALL)
