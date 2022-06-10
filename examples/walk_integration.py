import biorbd
import numpy as np

from bioptim import OdeSolver, CostType, RigidBodyDynamics
from bioptim import Solver, DefectType

from humanoid_2d import Humanoid2D, Integration, add_custom_plots, HumanoidOcp, HumanoidOcpMultiPhase


def torque_driven_dynamics(model: biorbd.Model, states: np.array, controls: np.array, params: np.array):
    q = states[: model.nbQ()]
    qdot = states[model.nbQ() :]
    tau = controls
    qddot = model.ForwardDynamicsConstraintsDirect(q, qdot, tau).to_array()
    return np.hstack((qdot, qddot))


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
    solv.set_maximum_iterations(1000)
    solv.set_linear_solver("ma57")
    solv.set_print_level(5)
    sol = humanoid.ocp.solve(solv)

    # --- Show results --- #
    print(sol.status)
    sol.print_cost()

    from bioptim import Shooting, SolutionIntegrator

    sol.integrate(
        shooting_type=Shooting.SINGLE_CONTINUOUS,
        keep_intermediate_points=False,
        merge_phases=False,
        continuous=True,
        integrator=SolutionIntegrator.SCIPY_DOP853,
    )
    print(sol.states["q"])

    integration = Integration(
        ocp=humanoid.ocp, solution=sol, state_keys=["q", "qdot"], control_keys=["tau"], function=torque_driven_dynamics
    )

    out = integration.integrate(
        shooting_type=Shooting.SINGLE_CONTINUOUS,
        keep_intermediate_points=False,
        merge_phases=False,
        continuous=True,
        integrator=SolutionIntegrator.SCIPY_DOP853,
    )
    print(out.states["q"])

    print(sol.states["q"] - out.states["q"])

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(sol.states["q"][0, :])
    plt.plot(out.states["q"][0, :])
    plt.show()
    # plot in red

    # ça plante pas à vérifier ;)


if __name__ == "__main__":
    main()
