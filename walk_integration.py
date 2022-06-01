import numpy as np

from humanoid_2d import Humanoid2D
from viz import add_custom_plots
from bioptim import OdeSolver, CostType, RigidBodyDynamics
from humanoid_ocp import HumanoidOcp
from humanoid_ocp_multiphase import HumanoidOcpMultiPhase
from bioptim import Solver, DefectType


def torque_driven_dynamics(model, states: np.array, controls: np.array):
    q = states[:model.nbQ()]
    qdot = states[model.nbQ():]
    tau = controls
    qddot = model.ForwardDynamics(q, qdot, tau).to_array()
    return np.vstack((qdot, qddot))


def main():
    n_shooting = 30
    ode_solver = OdeSolver.RK4()
    # ode_solver = OdeSolver.COLLOCATION()
    time = 0.3
    n_threads = 8
    # for human in Humanoid2D:
    human = Humanoid2D.HUMANOID_3DOF
    model_path = human
    print(human)
    # --- Solve the program --- #
    humanoid = HumanoidOcpMultiPhase(
        biorbd_model_path=model_path.value,
        phase_time=time,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        rigidbody_dynamics=RigidBodyDynamics.ODE,
        n_threads=n_threads,
        nb_phases=1,
    )

    add_custom_plots(humanoid.ocp)
    humanoid.ocp.add_plot_penalty(CostType.ALL)
    # humanoid.ocp.print()

    solv = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solv.set_maximum_iterations(0)
    solv.set_linear_solver("ma57")
    solv.set_print_level(5)
    sol = humanoid.ocp.solve(solv)

    # --- Show results --- #
    print(sol.status)
    sol.print_cost()

    from integration_function import Integration
    integration = Integration(ocp=humanoid.ocp,
                              solution=sol,
                              state_keys=["q", "qdot"],
                              control_keys=["tau"],
                              function=torque_driven_dynamics)
    q = integration.integrate()
    print(q.states["q"])

    # ça plante pas à vérifier ;)

if __name__ == "__main__":
    main()

