from humanoid_2d import Humanoid2D
from viz import add_custom_plots
from bioptim import OdeSolver, CostType, Transcription
from humanoid_ocp import HumanoidOcp
from bioptim import Solver


def main():
    n_shooting = 10
    ode_solver = OdeSolver.RK4(n_integration_steps=1)
    # ode_solver = OdeSolver.COLLOCATION()
    time = 0.3
    n_threads = 8
    # for human in Humanoid2D:
    human = Humanoid2D.HUMANOID_10DOF
    model_path = human
    print(human)
    # --- Solve the program --- #
    humanoid = HumanoidOcp(
        biorbd_model_path=model_path.value,
        phase_time=time,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        rigidbody_dynamics=Transcription.CONSTRAINT_FD,
        n_threads=n_threads,
    )

    add_custom_plots(humanoid.ocp)
    humanoid.ocp.add_plot_penalty(CostType.ALL)
    # humanoid.ocp.print()

    solv = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solv.set_print_level(5)
    sol = humanoid.ocp.solve(solv)

    # --- Show results --- #
    print(sol.status)
    sol.print_cost()
    sol.animate(n_frames=-1)
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
