import warnings

import biorbd_casadi as biorbd
import numpy as np
from scipy import interpolate
from bioptim import (
    OdeSolver,
    Node,
    OptimalControlProgram,
    ConstraintFcn,
    DynamicsFcn,
    ObjectiveFcn,
    QAndQDotBounds,
    QAndQDotAndQDDotBounds,
    ConstraintList,
    ObjectiveList,
    DynamicsList,
    Bounds,
    BoundsList,
    InitialGuessList,
    ControlType,
    Solver,
    InitialGuess,
    InterpolationType,
    PhaseTransitionList,
    PhaseTransitionFcn,
    RigidBodyDynamics,
)
from humanoid_initial_pose import set_initial_pose


class HumanoidOcp:
    def __init__(
        self,
        biorbd_model_path: str = None,
        n_shooting: int = 10,
        phase_time: float = 0.3,
        n_threads: int = 8,
        control_type: ControlType = ControlType.CONSTANT,
        ode_solver: OdeSolver = OdeSolver.COLLOCATION(),
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        step_length: float = 0.8,
        right_foot_location: np.array = np.zeros(3),
        use_sx: bool = False,
    ):
        self.biorbd_model_path = biorbd_model_path
        self.n_shooting = n_shooting
        self.phase_time = phase_time
        self.n_threads = n_threads
        self.control_type = control_type
        self.ode_solver = ode_solver
        self.rigidbody_dynamics = rigidbody_dynamics

        if biorbd_model_path is not None:
            self.biorbd_model = biorbd.Model(biorbd_model_path)
            self.n_shooting = n_shooting
            self.phase_time = phase_time

            self._set_head()
            self._set_knee()
            self._set_shoulder()

            self.n_q = self.biorbd_model.nbQ()
            self.n_qdot = self.biorbd_model.nbQdot()
            self.n_qddot = self.biorbd_model.nbQddot()
            self.n_qdddot = self.n_qddot
            self.n_tau = self.biorbd_model.nbGeneralizedTorque()

            self.tau_min, self.tau_init, self.tau_max = -500, 0, 500
            self.qddot_min, self.qddot_init, self.qddot_max = -1000, 0, 1000
            self.qdddot_min, self.qdddot_init, self.qdddot_max = -10000, 0, 10000

            self.right_foot_location = right_foot_location
            self.step_length = step_length
            self.initial_left_foot_location = right_foot_location - np.array([0, step_length / 2, 0])
            self.final_left_foot_location = right_foot_location + np.array([0, step_length / 2, 0])

            self.dynamics = DynamicsList()
            self.constraints = ConstraintList()
            self.objective_functions = ObjectiveList()
            self.phase_transitions = PhaseTransitionList()
            self.x_bounds = BoundsList()
            self.u_bounds = BoundsList()
            self.initial_states = []
            self.x_init = InitialGuessList()
            self.u_init = InitialGuessList()

            self.control_type = control_type
            self.control_nodes = Node.ALL if self.control_type == ControlType.LINEAR_CONTINUOUS else Node.ALL_SHOOTING

            self._set_dynamics()
            self._set_constraints()
            self._set_objective_functions()
            self._set_phase_transition()

            self._set_boundary_conditions()
            self._set_initial_guesses()

            self.ocp = OptimalControlProgram(
                self.biorbd_model,
                self.dynamics,
                self.n_shooting,
                self.phase_time,
                x_init=self.x_init,
                x_bounds=self.x_bounds,
                u_init=self.u_init,
                u_bounds=self.u_bounds,
                objective_functions=self.objective_functions,
                constraints=self.constraints,
                n_threads=n_threads,
                control_type=self.control_type,
                ode_solver=ode_solver,
                use_sx=use_sx,
            )

    def _set_head(self):
        self.has_head = False
        for i in range(self.biorbd_model.nbSegment()):
            seg = self.biorbd_model.segment(i)
            if seg.name().to_string() == "Head":
                self.has_head = True
                break

    def _set_knee(self):
        self.has_knee = False
        for i in range(self.biorbd_model.nbSegment()):
            seg = self.biorbd_model.segment(i)
            if seg.name().to_string() == "RShank":
                self.has_knee = True
                break

    def _set_shoulder(self):
        self.has_shoulder = False
        for i in range(self.biorbd_model.nbSegment()):
            seg = self.biorbd_model.segment(i)
            if seg.name().to_string() == "RArm":
                self.has_shoulder = True
                break

    def _set_dynamics(self):
        # warnings.warn("not implemented under this version of bioptim")
        self.dynamics.add(
            DynamicsFcn.TORQUE_DRIVEN, rigidbody_dynamics=self.rigidbody_dynamics, with_contact=True, phase=0
        )
        # self.dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=0)

    def _set_objective_functions(self):
        # --- Objective function --- #
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=0)

        idx_stability = [0, 1, 2]
        if self.has_head:
            idx_stability.append(3)

        # torso stability
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_QDDOT, phase=0, index=idx_stability, weight=0.01)

        # head stability
        if self.has_head:
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_QDDOT, derivative=True, phase=0, index=3, weight=0.01
            )
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, index=3, weight=0.01
            )

        # keep velocity CoM around 1.5 m/s
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, index=1, target=1.5, node=Node.START, weight=1000
        )
        self.objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, index=1, target=1.5, node=Node.END, weight=1000
        )

        # instead of phase transition along z
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_VELOCITY, index=2, weight=0.1)

        if (
            self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
            or self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
        ):
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, phase=0, key="qdddot", weight=1e-4)

    def _set_constraints(self):
        # --- Constraints --- #
        # Contact force in Z are positive
        self.constraints.add(
            ConstraintFcn.TRACK_CONTACT_FORCES, min_bound=0, max_bound=np.inf, node=Node.ALL, contact_index=1, phase=0
        )  # FP0 > 0 en Z

        # contact node at zero position and zero speed
        # node = Node.ALL if self.implicit_dynamics else Node.START
        node = Node.START
        self.constraints.add(
            ConstraintFcn.TRACK_MARKERS, node=node, target=self.right_foot_location, marker_index="RFoot", phase=0
        )
        self.constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, node=node, marker_index="RFoot", phase=0)
        # node = Node.END
        # self.constraints.add(
        #     ConstraintFcn.TRACK_MARKERS, node=node, target=self.right_foot_location, marker_index="RFoot", phase=0
        # )
        # self.constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, node=node, marker_index="RFoot", phase=0)

        # first and last step constraints
        self.constraints.add(
            ConstraintFcn.TRACK_MARKERS,
            target=self.initial_left_foot_location,
            node=Node.START,
            marker_index="LFoot",
            phase=0,
        )
        self.constraints.add(
            ConstraintFcn.TRACK_MARKERS,
            target=self.final_left_foot_location,
            node=Node.END,
            marker_index="LFoot",
            phase=0,
        )

        # Ensure lift of foot
        if self.has_knee:
            self.constraints.add(
                ConstraintFcn.TRACK_MARKERS,
                index=2,
                min_bound=0.05,
                max_bound=np.inf,
                node=Node.MID,
                marker_index="LFoot",
                phase=0,
            )

    def _set_phase_transition(self):
        idx = [0, 1, 2]
        idx = idx.append(3) if self.has_head else idx

        self.phase_transitions.add(PhaseTransitionFcn.CYCLIC, index=idx, weight=1000)

    def _set_boundary_conditions(self):
        self.x_bounds = BoundsList()
        self.x_bounds.add(
            bounds=QAndQDotAndQDDotBounds(self.biorbd_model)
            if self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
            or self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
            else QAndQDotBounds(self.biorbd_model)
        )
        nq = self.n_q

        self.x_bounds[0].max[2, :] = 0  # torso bended forward

        if self.has_head:
            self.x_bounds[0][nq + 3, 0] = 0  # head velocity zero at the beginning
            self.x_bounds[0][nq + 3, -1] = 0  # head velocity zero at the end

        if self.has_knee:
            self.x_bounds[0].min[nq - 2 : nq, 0] = -np.pi / 8  # driving knees

        # Supervised shoulders
        if self.has_shoulder:
            i = 1 if self.has_head else 0
            self.x_bounds[0][5 + i, 0] = -np.pi / 6
            self.x_bounds[0][6 + i, 0] = np.pi / 6
            self.x_bounds[0][5 + i, -1] = np.pi / 6
            self.x_bounds[0][6 + i, -1] = -np.pi / 6

            self.x_bounds[0][5 + i + nq, 0] = 0
            self.x_bounds[0][5 + i + nq, -1] = 0
            self.x_bounds[0][6 + i + nq, 0] = 0
            self.x_bounds[0][6 + i + nq, -1] = 0

            # Unsupervised arms not working trying another time with cyclic constraints
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
        if self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau
                + [self.qddot_min] * self.n_qddot
                + [self.qddot_min] * self.biorbd_model.nbContacts(),
                [self.tau_max] * self.n_tau
                + [self.qddot_max] * self.n_qddot
                + [self.qddot_max] * self.biorbd_model.nbContacts(),
            )
        elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau + [self.qddot_min] * self.n_qddot,
                [self.tau_max] * self.n_tau + [self.qddot_max] * self.n_qddot,
            )
        elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau
                + [self.qdddot_min] * self.n_qddot
                + [self.qddot_min] * self.biorbd_model.nbContacts(),
                [self.tau_max] * self.n_tau
                + [self.qdddot_max] * self.n_qddot
                + [self.qddot_max] * self.biorbd_model.nbContacts(),
            )
        elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau + [self.qdddot_min] * self.n_qddot,
                [self.tau_max] * self.n_tau + [self.qdddot_max] * self.n_qddot,
            )
        else:
            self.u_bounds.add([self.tau_min] * self.n_tau, [self.tau_max] * self.n_tau)
        # root is not actuated
        self.u_bounds[0][:3, :] = 0

    def _set_initial_guesses(self):
        """
        Set initial guess for the optimization problem.
        """

        # --- Initial guess --- #
        q0 = [0] * self.n_q
        # Torso over the floor and bent
        q0[1] = 0.8
        q0[2] = -np.pi / 6

        self.q0i = set_initial_pose(
            self.biorbd_model_path, np.array(q0), self.right_foot_location, self.initial_left_foot_location
        )
        self.q0end = set_initial_pose(
            self.biorbd_model_path, np.array(q0), self.right_foot_location, self.final_left_foot_location
        )

        qdot0 = [0] * self.n_qdot
        X0i = []
        X0i.extend(self.q0i)
        X0i.extend(qdot0)
        X0end = []
        X0end.extend(self.q0end)
        X0end.extend(qdot0)
        if (
            self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
            or self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
        ):
            X0i.extend([0] * self.n_qddot)
            X0end.extend([0] * self.n_qddot)
            # X0i.extend([0] * self.n_qddot + [0] * self.biorbd_model.nbContacts())
            # X0end.extend([0] * self.n_qddot + [0] * self.biorbd_model.nbContacts())

        x = np.linspace(0, self.phase_time, 2)
        y = np.array([X0i, X0end]).T
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.phase_time, self.n_shooting + 1)
        X0 = f(x_new)  # use interpolation function returned by `interp1d`

        self._set_initial_states(X0)
        self._set_initial_controls()

    def _set_initial_states(self, X0: np.array = None):
        if X0 is None:
            self.x_init.add([0] * (self.n_q + self.n_q))
        else:
            if X0.shape[1] != self.n_shooting + 1:
                X0 = self._interpolate_initial_states(X0)

            if not self.ode_solver.is_direct_shooting:
                n = self.ode_solver.polynomial_degree
                X0 = np.repeat(X0, n + 1, axis=1)
                X0 = X0[:, :-n]

            self.x_init.add(X0, interpolation=InterpolationType.EACH_FRAME)

    def _set_initial_controls(self, U0: np.array = None):
        if U0 is None:
            if self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
                self.u_init.add(
                    [self.tau_init] * self.n_tau
                    + [self.qddot_init] * self.n_qddot
                    + [5] * self.biorbd_model.nbContacts()
                )
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK:
                self.u_init.add(
                    [self.tau_init] * self.n_tau
                    + [self.qdddot_init] * self.n_qdddot
                    + [5] * self.biorbd_model.nbContacts()
                )
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK:
                self.u_init.add([self.tau_init] * self.n_tau + [self.qdddot_init] * self.n_qdddot)
            elif self.rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS:
                self.u_init.add([self.tau_init] * self.n_tau + [self.qddot_init] * self.n_qddot)
            else:
                self.u_init.add([self.tau_init] * self.n_tau)
        else:
            if U0.shape[1] != self.n_shooting:
                U0 = self._interpolate_initial_controls(U0)
            self.u_init.add(U0, interpolation=InterpolationType.EACH_FRAME)

    def _interpolate_initial_states(self, X0: np.array):
        print("interpolating initial states to match the number of shooting nodes")
        x = np.linspace(0, self.phase_time, X0.shape[1])
        y = X0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.phase_time, self.n_shooting + 1)
        y_new = f(x_new)  # use interpolation function returned by `interp1d`
        return y_new

    def _interpolate_initial_controls(self, U0: np.array):
        print("interpolating initial controls to match the number of shooting nodes")
        x = np.linspace(0, self.phase_time, U0.shape[1])
        y = U0
        f = interpolate.interp1d(x, y)
        x_new = np.linspace(0, self.phase_time, self.n_shooting)
        y_new = f(x_new)  # use interpolation function returned by `interp1d`
        return y_new
