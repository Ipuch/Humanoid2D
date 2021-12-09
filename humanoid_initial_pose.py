import biorbd as biorbd_eigen
from scipy import optimize
import numpy as np


def set_initial_pose(model_path: str, q0: np.ndarray, target_RFoot: np.ndarray, target_LFoot: np.ndarray):
    """

    :param target_LFoot:
    :param target_RFoot:
    :param model_path:
    :param q0:
    :param targetd:
    :param targetp:
    :return:
    """
    m = biorbd_eigen.Model(model_path)
    bound_min = []
    bound_max = []
    for i in range(m.nbSegment()):
        seg = m.segment(i)
        for r in seg.QRanges():
            bound_min.append(r.min())
            bound_max.append(r.max())
    bounds = (bound_min, bound_max)

    def objective_function(q, *args, **kwargs):
        markers = m.markers(q)
        out1 = np.linalg.norm(markers[0].to_array() - target_RFoot) ** 2
        out3 = np.linalg.norm(markers[-1].to_array() - target_LFoot) ** 2

        return out1 + out3

    pos = optimize.least_squares(
        objective_function,
        args=(m, target_RFoot, target_LFoot),
        x0=q0,
        bounds=bounds,
        verbose=2,
        method="trf",
        jac="3-point",
        ftol=2.22e-16,
        gtol=2.22e-16,
    )

    return pos.x
