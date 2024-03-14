# code based on https://gitlab.lrz.de/michaelkoelle/q-pol/-/blob/main/src/utils/env_util.py

from enum import Enum
from typing import Any

from gymnasium import Space
from gymnasium.spaces import Box, Discrete


class ActionType(Enum):
    """Action type of an environment"""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


def get_obs_dim(space: Space[Any]) -> int:
    """Get observation dimension"""
    if isinstance(space, Discrete):
        return int(space.n)
    elif isinstance(space, Box):
        return space.shape[0]
    else:
        raise NotImplementedError()


def get_act_dim(space: Space[Any]) -> int:
    """Get action dimension"""
    if isinstance(space, Discrete):
        return int(space.n)
    elif isinstance(space, Box):
        return space.shape[0]
    else:
        raise NotImplementedError()
