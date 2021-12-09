import numpy as np
from bioptim import (
    OdeSolver,
    Node,
    OptimalControlProgram,
    ConstraintFcn,
    ObjectiveFcn,
    DynamicsFcn,
    QAndQDotBounds,
    PhaseTransitionFcn,
    ConstraintList,
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InitialGuessList,
    PhaseTransitionList,
    BiMappingList,
    ControlType,
    Solver,
)

from .viz import add_custom_plots


class HumanoidOcp:
    def __init__(
        self,
        n_thread=8,
        control_type=ControlType.CONSTANT,
        ode_solver=OdeSolver.COLLOCATION(),
        update_obj=False,
        implicit_soft_contacts=False,
        implicit_dynamics=True,
    ):
        self.n_q, self.n_qdot, self.n_tau = -1, -1, -1

        self.implicit_soft_contacts = implicit_soft_contacts
        self.implicit_dynamics = implicit_dynamics
        self.mod = 2 if self.implicit_soft_contacts or self.implicit_dynamics else 1

        self.dynamics = DynamicsList()
        self.constraints = ConstraintList()
        self.objective_functions = ObjectiveList()
        self.x_bounds = BoundsList()
        self.u_bounds = BoundsList()
        self.initial_states = []
        self.x_init = InitialGuessList()
        self.u_init = InitialGuessList()

        self.jumper = jumper
        self.control_type = control_type
        self.control_nodes = Node.ALL if self.control_type == ControlType.LINEAR_CONTINUOUS else Node.ALL_SHOOTING

        self._set_dimensions_and_mapping()
        self._set_initial_states()

        self._set_dynamics()
        self._set_constraints()
        self._set_objective_functions(update_obj)

        self._set_boundary_conditions()
        self._set_initial_guesses()

        self.ocp = OptimalControlProgram(
            self.jumper.model,
            self.dynamics,
            self.jumper.n_shoot,
            self.jumper.phase_time,
            x_init=self.x_init,
            x_bounds=self.x_bounds,
            u_init=self.u_init,
            u_bounds=self.u_bounds,
            objective_functions=self.objective_functions,
            constraints=self.constraints,
            n_threads=n_thread,
            control_type=self.control_type,
            ode_solver=ode_solver,
        )

    def _set_initial_states(self):
        initial_pose = self.jumper.find_initial_root_pose()
        # self.jumper.show(initial_pose)
        initial_velocity = np.array([self.jumper.initial_velocity]).T
        self.initial_states = np.concatenate((initial_pose, initial_velocity))

    def _set_dimensions_and_mapping(self):
        self.n_q = self.jumper.model.nbQ()
        self.n_qdot = self.jumper.model.nbQdot()
        self.n_tau = self.jumper.model.nbGeneralizedTorque()

    def _set_dynamics(self):
        self.dynamics.add(
            DynamicsFcn.TORQUE_DRIVEN,
            implicit_soft_contacts=self.implicit_soft_contacts,
            implicit_dynamics=self.implicit_dynamics,
        )

    def _set_constraints(self):
        pass

    def _set_objective_functions(self, update_obj):
        # Maximize the jump height
        if not update_obj:
            #
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_VELOCITY, weight=100, quadratic=True)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, quadratic=True)
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, derivative=True, key="tau", weight=2, quadratic=True
            )
            # self.objective_functions.add(ObjectiveFcn.Lagrange.Mini, derivative=True, key="tau", weight=2,
            #                              quadratic=True)
            # self.objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", weight=50, quadratic=True,
            # node=Node.END)
            # self.objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, weight=-100, quadratic=False)
        else:
            self.objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, weight=-100, quadratic=False, index=2
            )
            # self.objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, weight=-100, quadratic=False)
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, derivative=True, key="tau", weight=2, quadratic=True
            )
            self.objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, weight=-100, quadratic=False, index=2, node=Node.PENULTIMATE
            )

        # Minimize time of the phase
        if self.jumper.time_min != self.jumper.time_max:
            self.objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_TIME,
                weight=0.1,
                min_bound=self.jumper.time_min,
                max_bound=self.jumper.time_max,
            )

    def _set_boundary_conditions(self):
        # Path constraints
        self.x_bounds.add(bounds=QAndQDotBounds(self.jumper.model))
        self.u_bounds.add(
            [-self.jumper.tau_constant_bound] * self.n_tau * self.mod,
            [self.jumper.tau_constant_bound] * self.n_tau * self.mod,
        )
        self.u_bounds[0][: self.jumper.model.nbRoot(), :] = 0

        # Enforce the initial pose and velocity
        self.x_bounds[0].max[:, 0] = self.initial_states[:, 0] + 1e-4
        self.x_bounds[0].min[:, 0] = self.initial_states[:, 0] - 1e-4

        # self.x_bounds[0].max[self.n_q:, -1] = + 1e-3
        # self.x_bounds[0].min[self.n_q:, -1] = - 1e-3

        # self.x_bounds[0][:, 0] = self.initial_states[:, 0]

    def _set_initial_guesses(self):
        self.x_init.add(self.initial_states)
        self.u_init.add([0] * self.n_tau * self.mod)

    def solve(
        self,
        limit_memory_max_iter,
        exact_max_iter,
        load_path=None,
        force_no_graph=False,
        linear_solver="mumps",
        sol=None,
    ):
        # Run optimizations
        if not force_no_graph:
            add_custom_plots(self.ocp, self)

        if load_path:
            _, sol = OptimalControlProgram.load(load_path)
            return sol
        else:
            solver = Solver.IPOPT()
            solver.set_linear_solver(linear_solver)
            # solver.set_print_level(0)

            if limit_memory_max_iter > 0 and sol is None:
                solver.set_hessian_approximation("limited-memory")
                solver.set_maximum_iterations(limit_memory_max_iter)
                solver.show_online_optim = exact_max_iter == 0 and not force_no_graph
                solver.set_convergence_tolerance(1e-2)
                sol = self.ocp.solve(solver)

            if (limit_memory_max_iter > 0 or sol is not None) and exact_max_iter > 0:
                self.ocp.set_warm_start(sol)

            if exact_max_iter > 0:
                solver.set_hessian_approximation("exact")
                solver.set_maximum_iterations(exact_max_iter)
                solver.set_warm_start_options()
                solver.show_online_optim = not force_no_graph
                solver.set_convergence_tolerance(1e-2)
                sol = self.ocp.solve(solver)

            return sol
