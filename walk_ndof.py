from humanoid_2d import Humanoid2D
from viz import add_custom_plots
from bioptim import OdeSolver, CostType, Transcription
from humanoid_ocp import HumanoidOcp
from humanoid_ocp_multiphase import HumanoidOcpMultiPhase
from bioptim import Solver


def main():
    n_shooting = 30
    ode_solver = OdeSolver.RK4(n_integration_steps=5)
    # ode_solver = OdeSolver.COLLOCATION()
    time = 0.3
    n_threads = 8
    # for human in Humanoid2D:
    human = Humanoid2D.HUMANOID_10DOF
    model_path = human
    print(human)
    # --- Solve the program --- #
    humanoid = HumanoidOcpMultiPhase(
        biorbd_model_path=model_path.value,
        phase_time=time,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        rigidbody_dynamics=Transcription.ODE,
        n_threads=n_threads,
        nb_phases=2,
    )

    add_custom_plots(humanoid.ocp)
    humanoid.ocp.add_plot_penalty(CostType.ALL)
    # humanoid.ocp.print()

    solv = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solv.set_maximum_iterations(1000)
    solv.set_linear_solver("ma57")
    solv.set_print_level(5)
    sol = humanoid.ocp.solve(solv)

    # --- Show results --- #
    print(sol.status)
    sol.print_cost()
    sol.animate(n_frames=0)
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
