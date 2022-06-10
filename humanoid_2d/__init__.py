"""
humanoid_2d is a package for the Humanoid2D model.
-------------------------------------------------------------------------


# --- The main optimal control programs --- #
HumanoidOcp

HumanoidOcpMultiPhase

"""

from .models.humanoid_2d import Humanoid2D
from .ocp.viz import add_custom_plots
from .ocp.humanoid_ocp import HumanoidOcp
from .ocp.humanoid_ocp_multiphase import HumanoidOcpMultiPhase
from .initial_guess.humanoid_initial_pose import set_initial_pose

from .bioptim_plugin.integration_function import Integration


