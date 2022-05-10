import numpy as np
import biorbd_casadi as biorbd
from casadi import vertcat
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    DynamicsList,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintList,
    ConstraintFcn,
    InterpolationType,
    Node,
    BoundsList,
    OdeSolver,
    Solver,
    CostType,
    PhaseTransitionList,
    PhaseTransitionFcn,
    PhaseTransition,
    OptimizationVariableList,
)


def anti_symmetric_cyclic_transition(
    transition: PhaseTransition,
    state_pre: OptimizationVariableList,
    state_post: OptimizationVariableList,
    first_index: int,
    second_index: int,
):
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition. If coef=1, then this function mimics the PhaseTransitionFcn.CONTINUOUS

    Parameters
    ----------
    transition: PhaseTransition
        The ...
    state_pre: MX
        The states at the end of a phase
    state_post: MX
        The state at the beginning of the next phase
    first_index: int
        first state to be concerned
    second_index: int
        second state to be concerned

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    # states_mapping can be defined in PhaseTransitionList. For this particular example, one could simply ignore the
    # mapping stuff (it is merely for the sake of example how to use the mappings)
    states_pre = transition.states_mapping.to_second.map(state_pre.cx_end)
    states_post = transition.states_mapping.to_first.map(state_post.cx)

    first_constraint = states_pre[first_index] - states_post[second_index]
    second_constraint = states_pre[second_index] - states_post[first_index]

    return vertcat(first_constraint, second_constraint)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = False,
    n_threads: int = 1,
    implicit_dynamics: bool = False,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolver = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)
    implicit_dynamics: bool
        implicit
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    model = biorbd.Model(biorbd_model_path)
    n_q = model.nbQ()
    n_qdot = model.nbQdot()
    n_tau = model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -400, 400, 0

    # --- Dynamics --- #
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=0)

    # --- Objective function --- #
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=0, weight=100)

    # torso stability
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_QDDOT, phase=0, index=[0, 1, 2], weight=0.01)
    # head stability
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_QDDOT, derivative=True, phase=0, index=3, weight=0.01)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, index=3, weight=0.01)

    # keep velocity CoM around 1.5 m/s
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, index=1, target=1.5, node=Node.START, weight=1000)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, index=1, target=1.5, node=Node.END, weight=1000)

    # instead of phase transition
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_VELOCITY, index=2, weight=0.1)

    # --- Constraints --- #
    constraints = ConstraintList()
    # Contact force in Z are positive
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES, min_bound=0, max_bound=np.inf, node=Node.ALL, contact_index=1, phase=0
    )  # FP0 > 0 en Z

    # contact node at zero position and zero speed
    constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.START, marker_index="RFoot", phase=0)
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, node=Node.START, marker_index="RFoot", phase=0)

    # first and last step constraints
    constraints.add(
        ConstraintFcn.TRACK_MARKERS, target=np.array([0, -0.4, 0]), node=Node.START, marker_index="LFoot", phase=0
    )
    constraints.add(
        ConstraintFcn.TRACK_MARKERS, target=np.array([0, 0.4, 0]), node=Node.END, marker_index="LFoot", phase=0
    )
    # Ensure lift of foot
    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        index=2,
        min_bound=0.05,
        max_bound=np.inf,
        node=Node.MID,
        marker_index="LFoot",
        phase=0,
    )

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CYCLIC, index=2, weight=10)
    i = 0
    ii = 2
    phase_transitions.add(anti_symmetric_cyclic_transition, first_index=3 + i, second_index=4 + i, phase_pre_idx=0)
    phase_transitions.add(
        anti_symmetric_cyclic_transition, first_index=5 + i + ii, second_index=6 + i + ii, phase_pre_idx=0
    )

    phase_transitions.add(
        anti_symmetric_cyclic_transition,
        first_index=3 + i + n_q,
        second_index=4 + i + n_q,
        phase_pre_idx=0,
        weight=0.01,
    )
    phase_transitions.add(
        anti_symmetric_cyclic_transition,
        first_index=5 + i + ii + n_q,
        second_index=6 + i + ii + n_q,
        phase_pre_idx=0,
        weight=0.01,
    )

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(model))

    # x_bounds[0][n_q + 3, 0] = 0  # head velocity zero at the beginning
    x_bounds[0].max[2, :] = 0  # torso bended forward

    # Supervised shoulders
    x_bounds[0][5 + i, 0] = -np.pi / 8
    x_bounds[0][6 + i, 0] = np.pi / 8
    x_bounds[0][5 + i, -1] = np.pi / 8
    x_bounds[0][6 + i, -1] = -np.pi / 8

    x_bounds[0][5 + i + n_q, 0] = 0
    x_bounds[0][5 + i + n_q, -1] = 0
    x_bounds[0][6 + i + n_q, 0] = 0
    x_bounds[0][6 + i + n_q, -1] = 0

    # Unsupervised arms not working trying another time with cyclic constraints
    # x_bounds[0][n_q + 3, 0] = 0  # head velocity zero at the beginning
    # x_bounds[0].max[2, :] = 0  # torso bended forward
    #
    # x_bounds[0].max[5, 0] = -1e-5  # position is negative at start
    # x_bounds[0].min[6, 0] = 1e-5  # position is positive at start
    #
    # x_bounds[0].min[5, -1] = 1e-5  # position is positive at the end
    # x_bounds[0].max[6, -1] = -1e-5  # position is negative at the end
    #
    # x_bounds[0][n_q + 5, [0, -1]] = 0  # velocity of shoulders zero at begining and end
    # x_bounds[0][n_q + 6, [0, -1]] = 0  # velocity of shoulders zero at begining and end
    # x_bounds[0].max[n_q + 6, 1] = -1e-5  # velocity of left shoulder negative
    # x_bounds[0].min[n_q + 6, 1] = -5  # velocity of left shoulder negative
    # x_bounds[0].min[n_q + 5, 1] = 1e-5 # velocity of right shoulder positive
    # x_bounds[0].max[n_q + 5, 1] = 5  # velocity of right shoulder positive

    u_bounds = BoundsList()
    u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)
    # root is no actuated
    u_bounds[0][:3, :] = 0

    # --- Initial guess --- #
    q0 = [0] * n_q
    # Torso over the floor and bended
    q0[1] = 0.8
    q0[2] = -3.14 / 6
    qdot0 = [0] * n_qdot
    X0 = []
    X0.extend(q0)
    X0.extend(qdot0)
    x_init = InitialGuess(X0, interpolation=InterpolationType.CONSTANT)
    u_init = InitialGuess([tau_init] * n_tau)

    return OptimalControlProgram(
        biorbd_model=model,
        dynamics=dynamics,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        phase_time=final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        phase_transitions=phase_transitions,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def main():
    model_path = "models/Humanoid7Dof.bioMod"
    n_shooting = 10
    ode_solver = OdeSolver.RK4(n_integration_steps=5)
    # ode_solver = OdeSolver.COLLOCATION()
    time = 0.3
    n_threads = 8
    # --- Solve the program --- #
    ocp = prepare_ocp(
        biorbd_model_path=model_path,
        final_time=time,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        implicit_dynamics=False,
        n_threads=n_threads,
    )
    # ocp.add_plot_penalty(CostType.ALL)

    # Plot CoM pos and velocity
    for i, nlp in enumerate(ocp.nlp):
        ocp.add_plot(
            "CoM", lambda t, x, u, p: plot_com(x, nlp), phase=i, legend=["CoMy", "Comz", "CoM_doty", "CoM_dotz"]
        )

    solv = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    sol = ocp.solve(solv)

    # --- Show results --- #
    sol.print()
    sol.animate()
    # sol.graphs(show_bounds=True)

    print("verify phase transitions")
    i = 0
    ii = 2

    print(sol.states["q"][3 + i, 0] - sol.states["q"][4 + i, -1] * 180 / np.pi)
    print(sol.states["q"][4 + i, 0] - sol.states["q"][3 + i, -1] * 180 / np.pi)
    print(sol.states["qdot"][3 + ii, 0] - sol.states["qdot"][4 + i, -1] * 180 / np.pi)
    print(sol.states["qdot"][4 + i, 0] - sol.states["qdot"][3 + i, -1] * 180 / np.pi)

    print(sol.states["q"][5 + i + ii, 0] - sol.states["q"][6 + i + ii, -1] * 180 / np.pi)
    print(sol.states["q"][6 + i + ii, 0] - sol.states["q"][5 + i + ii, -1] * 180 / np.pi)
    print(sol.states["qdot"][5 + i + ii, 0] - sol.states["qdot"][6 + i + ii, -1] * 180 / np.pi)
    print(sol.states["qdot"][6 + i + ii, 0] - sol.states["qdot"][5 + i + ii, -1] * 180 / np.pi)


def plot_com(x, nlp):
    com_func = biorbd.to_casadi_func("CoMPlot", nlp.model.CoM, nlp.states["q"].mx, expand=False)
    com_dot_func = biorbd.to_casadi_func(
        "Compute_CoM", nlp.model.CoMdot, nlp.states["q"].mx, nlp.states["qdot"].mx, expand=False
    )
    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])

    return np.concatenate((np.array(com_func(q)[1:, :]), np.array(com_dot_func(q, qdot)[1:, :])))


if __name__ == "__main__":
    main()
