from typing import Any, Dict, Sequence, Tuple, Union, List

from dial_mpc.envs.unitree_go2_env import (
    UnitreeGo2EnvConfig,
)

_configs = {
    "unitree_go2_walk": UnitreeGo2EnvConfig,
}


def register_config(name: str, config: Any):
    _configs[name] = config


def get_config(name: str) -> Any:
    return _configs[name]
