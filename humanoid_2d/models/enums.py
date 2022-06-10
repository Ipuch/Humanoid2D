from enum import Enum, IntEnum
from pathlib import Path


class Humanoid2D(Enum):
    HUMANOID_3DOF = (Path(__file__).parent.__str__() + "/Humanoid3Dof.bioMod",)
    HUMANOID_4DOF = (Path(__file__).parent.__str__() + "/Humanoid4Dof.bioMod",)
    HUMANOID_5DOF = (Path(__file__).parent.__str__() + "/Humanoid5Dof.bioMod",)
    HUMANOID_6DOF = (Path(__file__).parent.__str__() + "/Humanoid6Dof.bioMod", "/Humanoid6Dof_left_contact.bioMod")
    HUMANOID_7DOF = (Path(__file__).parent.__str__() + "/Humanoid7Dof.bioMod",)
    HUMANOID_8DOF = (Path(__file__).parent.__str__() + "/Humanoid8Dof.bioMod",)
    HUMANOID_9DOF = (Path(__file__).parent.__str__() + "/Humanoid9Dof.bioMod",)
    HUMANOID_10DOF = (Path(__file__).parent.__str__() + "/Humanoid10Dof.bioMod", "/Humanoid10Dof_left_contact.bioMod")
