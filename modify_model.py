import biorbd_casadi as biorbd

my_path = "models/Humanoid10Dof.bioMod"
biorbd_model = biorbd.Model(my_path)
biorbd_model.segment(0)
