"""
This script runs the miller optimal control problem with a given set of parameters and save the results.
The main function is used in main_comparison.py and main_convergence.py. to run the different Miller optimal control problem.
"""
import numpy as np
import pickle
from time import time

import biorbd
from bioptim import Solver, Shooting, RigidBodyDynamics, Shooting, SolutionIntegrator
from humanoid_2d import HumanoidOcpMultiPhase, Integration


def torque_driven_dynamics(model: biorbd.Model, states: np.array, controls: np.array, params: np.array):
    q = states[: model.nbQ()]
    qdot = states[model.nbQ() :]
    tau = controls
    qddot = model.ForwardDynamicsConstraintsDirect(q, qdot, tau).to_array()
    return np.hstack((qdot, qddot))


def main(args: list = None):
    """
    Main function for the miller_run.py script.
    It runs the optimization and saves the results of a Miller Optimal Control Problem.

    Parameters
    ----------
    args : list
        List of arguments containing the following:
        args[0] : biorbd_model_path
            Path to the biorbd model.
        args[1] : i_rand
            Random seed.
        args[2] : n_shooting
            Number of shooting nodes.
        args[3] : dynamics_type (RigidBodyDynamics)
            Type of dynamics to use such as RigidBodyDynamics.ODE or RigidBodyDynamics.DAE_INVERSE_DYNAMICS, ...
        args[4] : ode_solver
            Type of ode solver to use such as OdeSolver.RK4, OdeSolver.RK2, ...
        args[5] : nstep
            Number of steps for the ode solver.
        args[6] : n_threads
            Number of threads to use.
        args[7] : out_path_raw
            Path to save the raw results.
    """
    if args:
        biorbd_model_path = args[0]
        ode_solver = args[1]
        n_shooting = args[2]
        n_threads = args[3]
        dynamics_type = args[4]
        n_phases = args[5]
        out_path_raw = args[6]
        i_rand = args[7]
    else:
        biorbd_model_path = args[0]
        ode_solver = args[1]
        n_shooting = args[2]
        n_threads = args[3]
        dynamics_type = args[4]
        n_phases = args[5]
        out_path_raw = args[6]
        i_rand = args[7]

    # to handle the random multi-start of the ocp
    np.random.seed(i_rand)
    # --- Solve the program --- #
    humanoid_ocp = HumanoidOcpMultiPhase(
        biorbd_model_path=biorbd_model_path.value,
        nb_phases=n_phases,
        rigidbody_dynamics=dynamics_type,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        n_threads=n_threads,
    )
    str_ode_solver = ode_solver.__str__().replace("\n", "_").replace(" ", "_")
    filename = f"humanoid_irand{i_rand}_{n_shooting}_{str_ode_solver}"
    outpath = f"{out_path_raw}/" + filename

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(10000)
    solver.set_print_level(5)
    solver.set_linear_solver("ma57")

    print(f"##########################################################")
    print(
        f"Solving dynamics_type={dynamics_type}, i_rand={i_rand},"
        f"n_shooting={n_shooting}\n"
        f"ode_solver={str_ode_solver}, n_threads={n_threads}"
    )
    print(f"##########################################################")

    # --- time to solve --- #
    tic = time()
    sol = humanoid_ocp.ocp.solve(solver)
    toc = time() - tic

    states = sol.states["all"]
    controls = sol.controls["all"]
    parameters = sol.parameters["all"]

    sol.print_cost()

    print(f"##########################################################")
    print(
        f"Time to solve dynamics_type={dynamics_type}, i_rand={i_rand},"
        f"n_shooting={n_shooting}\n"
        f"ode_solver={str_ode_solver}, n_threads={n_threads}"
        f"\n {toc}sec\n"
    )
    print(f"##########################################################")

    # --- Save the results --- #

    # integrer la dynamique direct

    integration = Integration(
        ocp=humanoid_ocp.ocp,
        solution=sol,
        state_keys=["q", "qdot"],
        control_keys=["tau"],
        function=torque_driven_dynamics,
    )

    out = integration.integrate(
        shooting_type=Shooting.SINGLE_CONTINUOUS,
        keep_intermediate_points=False,
        merge_phases=False,
        continuous=True,
        integrator=SolutionIntegrator.SCIPY_DOP853,
    )

    # sol_integrated = sol.integrate(shooting_type=Shooting.MULTIPLE, keep_intermediate_points=False, merge_phases=False, continuous=False)

    f = open(f"{outpath}.pckl", "wb")
    data = {
        "model_path": biorbd_model_path,
        "phase_time": humanoid_ocp.phase_time,
        "irand": i_rand,
        "computation_time": toc,
        "cost": sol.cost,
        "detailed_cost": sol.detailed_cost,
        "iterations": sol.iterations,
        "status": sol.status,
        "states": sol.states,
        "controls": sol.controls,
        "parameters": sol.parameters,
        "time": out.time_vector,
        "dynamics_type": dynamics_type,
        "ode_solver": ode_solver,
        "q": sol.states_no_intermediate["q"],
        "qdot": sol.states_no_intermediate["qdot"],
        "q_integrated": out.states["q"],
        "qdot_integrated": out.states["qdot"],
        # "qddot_integrated": out.states["qdot"],
        "n_shooting": n_shooting,
        "n_theads": n_threads,
    }
    pickle.dump(data, f)
    f.close()

    humanoid_ocp.ocp.save(sol, f"{outpath}.bo")


if __name__ == "__main__":
    main()
