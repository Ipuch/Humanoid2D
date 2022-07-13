"""
Todo
"""
import os
from utils import generate_calls, run_pool, run_the_missing_ones
from humanoid_2d import Humanoid2D
from multiprocessing import Pool, cpu_count
from datetime import date
from bioptim import OdeSolver, RigidBodyDynamics
from pathlib import Path


def main():
    Date = date.today()
    Date = Date.strftime("%d-%m-%y")

    out_path_raw = Path(Path(__file__).parent.__str__() + f"/../../Humanoid2D_results/raw_{Date}")
    try:
        os.mkdir(out_path_raw)
    except:
        print("../Humanoid2D_results/raw_" + Date + " is already created ")

    cpu_number = cpu_count()
    n_thread = 8
    param = dict(
        model_str=[Humanoid2D.HUMANOID_10DOF],
        # ode_solver=[OdeSolver.CVODES()],
        ode_solver=[OdeSolver.RK4(n_integration_steps=1), OdeSolver.IRK(), OdeSolver.COLLOCATION(), OdeSolver.CVODES()],
        n_shooting=[30],
        n_thread=[n_thread],
        # dynamic_type=[RigidBodyDynamics.ODE, RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK],
        dynamic_type=[RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK],
        n_phases=[1],
        out_path=[out_path_raw.absolute().__str__()],
    )
    calls = int(1)

    my_calls = generate_calls(
        calls,
        param,
    )

    my_pool_number = int(cpu_number / n_thread)
    from run_humanoid import main as run_humanoid
    # run_humanoid(my_calls[0])
    run_pool(my_calls, my_pool_number)

    # run_the_missing_ones(
    #     out_path_raw, Date, n_shooting, ode_solver, nsteps, n_thread, model_str, my_pool_number
    # )


if __name__ == "__main__":
    main()
