from typing import Any, Union
import numpy as np

import biorbd
from bioptim import Solution


def compute_error_single_shooting(
    time: Union[np.ndarray, list],
    n_shooting: int,
    model: biorbd.Model,
    q: np.ndarray,
    q_integrated: np.ndarray,
    duration: float = None,
):
    """
    Compute the error between the solution of the OCP and the solution of the integrated OCP

    Parameters
    ----------
    time : np.ndarray
        Time vector
    n_shooting : int
        Number of shooting points
    model : biorbd.Model
        Model
    q : np.ndarray
        ocp generalized coordinates
    q_integrated : np.ndarray
        integrated generalized coordinates
    duration: float
        The duration to report the error in states btween the two solutions

    Returns
    -------
    The error between the two solutions
    :tuple
    """

    duration = time[-1] if duration is None else duration

    if time[-1] < duration:
        raise ValueError(
            f"Single shooting integration duration must be smaller than ocp duration :{time[-1]} s"
        )

    # get the index of translation and rotation dof
    trans_idx = []
    rot_idx = []
    for i in range(model.nbQ()):
        if model.nameDof()[i].to_string()[-4:-1] == "Rot":
            rot_idx += [i]
        else:
            trans_idx += [i]
    rot_idx = np.array(rot_idx)
    trans_idx = np.array(trans_idx)

    sn_1s = int(n_shooting / time[-1] * duration)  # shooting node at {duration} second
    single_shoot_error_r = (
        rmse(q[rot_idx, sn_1s], q_integrated[rot_idx, sn_1s]) * 180 / np.pi if len(rot_idx) > 0 else np.nan
    )

    single_shoot_error_t = (
        (rmse(q[trans_idx, sn_1s], q_integrated[trans_idx, sn_1s]) / 1000) if len(trans_idx) > 0 else np.nan
    )

    return (
        single_shoot_error_t,
        single_shoot_error_r,
    )


def stack_states(states: list[dict], key: str = "q"):
    """
    Stack the controls in one vector

    Parameters
    ----------
    states : list[dict]
        List of dictionaries containing the states
    key : str
        Key of the states to stack such as "q" or "qdot"
    """
    the_tuple = (s[key][:, :-1] if i < len(states) else s[key][:, :] for i, s in enumerate(states))
    return np.hstack(the_tuple)


def stack_controls(controls: list[dict], key: str = "tau"):
    """
    Stack the controls in one vector

    Parameters
    ----------
    controls : list[dict]
        List of dictionaries containing the controls
    key : str
        Key of the controls to stack such as "tau" or "qddot"
    """
    the_tuple = (c[key][:, :-1] if i < len(controls) else c[key][:, :] for i, c in enumerate(controls))
    return np.hstack(the_tuple)


def define_time(time: list, n_shooting: tuple):
    """
    Create the time vector

    Parameters
    ----------
    time : list
        List of duration of each phase of the simulation
    n_shooting : tuple
        Number of shooting points for each phase
    """
    the_tuple = (
        np.linspace(0, float(time[i]) - 1 / n_shooting[i] * float(time[i]), n_shooting[i])
        if i < len(time)
        else np.linspace(float(time[i]), float(time[i]) + float(time[i + 1]), n_shooting[i] + 1)
        for i, t in enumerate(time)
    )
    return np.hstack(the_tuple)


def rmse(predictions, targets) -> float:
    """
    Compute the Root Mean Square Error

    Parameters
    ----------
    predictions : numpy.array
        Predictions
    targets : numpy.array
        Targets

    Returns
    -------
    rmse : float
        Root Mean Square Error
    """
    return np.sqrt(((predictions - targets) ** 2).mean())
