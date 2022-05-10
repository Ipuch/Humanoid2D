from humanoid_2d import Humanoid2D
from viz import add_custom_plots
from bioptim import OdeSolver, SolutionIntegrator
from humanoid_ocp import HumanoidOcp
from bioptim import Solver
from Comparison import integrate_sol, compute_error_single_shooting


def main():
    n_shooting = 10
    ode_solver = OdeSolver.COLLOCATION()
    time = 0.3
    n_threads = 8
    for human in Humanoid2D:
        model_path = human
        print(human)
        # --- Solve the program --- #
        humanoid = HumanoidOcp(
            biorbd_model_path=model_path.value,
            phase_time=time,
            n_shooting=n_shooting,
            ode_solver=ode_solver,
            implicit_dynamics=False,
            n_threads=n_threads,
        )

        add_custom_plots(humanoid.ocp)
        # humanoid.ocp.print()

        solv = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
        solv.set_print_level(5)
        solv.set_maximum_iterations(0)
        sol = humanoid.ocp.solve(solv)

        # --- Show results --- #
        print(sol.status)
        # sol.print()
        # sol.animate()
        # sol.graphs(show_bounds=True)

        compute_error_single_shooting(sol, duration=time, integrator=SolutionIntegrator.DEFAULT)


if __name__ == "__main__":
    main()
