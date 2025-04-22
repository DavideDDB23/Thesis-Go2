import importlib.resources
import os
import numpy as np
import torch


def get_example_path(example_name):
    with importlib.resources.path(f"dial_mpc.examples", example_name) as path:
        return path


def load_dataclass_from_dict(dataclass, data_dict, convert_list_to_array=False):
    keys = dataclass.__dataclass_fields__.keys() & data_dict.keys()
    kwargs = {key: data_dict[key] for key in keys}
    if convert_list_to_array:
        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = torch.tensor(value)
    return dataclass(**kwargs)
