"""
Todo
"""
import os
from multiprocess_utils import generate_calls, run_pool, run_the_missing_ones
from humanoid_2d import Humanoid2D
from multiprocessing import Pool, cpu_count
from datetime import date
from bioptim import OdeSolver, RigidBodyDynamics

Date = date.today()
Date = Date.strftime("%d-%m-%y")

out_path_raw = "../Humanoid2D_results/raw_" + Date
try:
    os.mkdir(out_path_raw)
except:
    print("../Humanoid2D_results/raw_" + Date + " is already created ")

cpu_number = cpu_count()
n_thread = 8
param = dict(
    model_str=[Humanoid2D.HUMANOID_10DOF],
    ode_solver=[OdeSolver.RK4(n_integration_steps=1),
                OdeSolver.CVODES,
                OdeSolver.IRK,
                OdeSolver.COLLOCATION],
    n_shooting=[30],
    n_thread=[n_thread],
    dynamic_type=[RigidBodyDynamics.ODE],
    n_phases=[1],
)
calls = 5


my_calls = generate_calls(
    calls,
    param,
)

my_pool_number = int(cpu_number / n_thread)
run_pool(my_calls, my_pool_number, out_path=out_path_raw)

# run_the_missing_ones(
#     out_path_raw, Date, n_shooting, ode_solver, nsteps, n_thread, model_str, my_pool_number
# )
