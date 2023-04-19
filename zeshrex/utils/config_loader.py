from pathlib import Path
from types import SimpleNamespace
from typing import Union, Callable

import yaml

from zeshrex.utils.misc import convert_dict_to_namespace


def load_yaml_config(path: Union[Path, str], convert_to_namespace: bool = False) -> Union[dict, SimpleNamespace]:
    """
    Loads YAML config file.

    :param path: Path to the config file to load.
    :param convert_to_namespace: Whether to convert configs dictionary into SimpleNamespace object to use dot notation
                                 for accessing a variable.
    :return: Dictionary (of dictionaries, etc., possibly) with configs provided in YAML file.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} was not found!")
    if path.suffix not in ('.yaml', '.yml'):
        raise ValueError(f"Provided file {path} is not a YAML config file!")

    with open(path, 'r') as stream:
        conf = yaml.safe_load(stream)

    if convert_to_namespace:
        conf = convert_dict_to_namespace(conf)

    return conf


def print_configs(cfg: Union[dict, SimpleNamespace],
                  print_function: Callable = print) -> None:
    """
    Prints configs to the output stream.

    :param cfg: Configs to print. Configs object can be whether dictionary or SimpleNamespace.
    :param print_function: Defines a function that will be used for output. It can be simple `print`
                           function (default), logger with stream and file handlers, etc.
    :return: None
    """
    print_function("--------------------------------------------------")
    print_function("Script was initialized with the following configs:")

    def convert_cfg_to_dict(raw_cfg: Union[dict, SimpleNamespace]) -> dict:
        if isinstance(raw_cfg, SimpleNamespace):
            cfg_dict = raw_cfg.__dict__
        elif isinstance(raw_cfg, dict):
            cfg_dict = raw_cfg
        else:
            raise ValueError("Configs object has unsupported type!")
        return cfg_dict

    def print_cfg_recursively(cfg_to_print: Union[dict, SimpleNamespace], pfunc: Callable, indent: int = 0):
        cfg_dict_to_print = convert_cfg_to_dict(cfg_to_print)
        max_key_length: int = max([len(s) for s in cfg_dict_to_print.keys()])
        for k, v in cfg_dict_to_print.items():
            if isinstance(v, dict) or isinstance(v, SimpleNamespace):
                pfunc("{:{align_outer}{width_outer}} {:{align_inner}{width_inner}} :".format(
                    '|', k,
                    align_outer='>', width_outer=f'{indent}',
                    align_inner='>', width_inner=f'{max_key_length}')
                )
                print_cfg_recursively(v, pfunc, indent=max_key_length + indent + 4)
            else:
                pfunc("{:{align_outer}{width_outer}} {:{align_inner}{width_inner}} : {}".format(
                    '|', k, v,
                    align_outer='>', width_outer=f'{indent}',
                    align_inner='>', width_inner=f'{max_key_length}')
                )

    print_cfg_recursively(cfg, print_function)

    print_function("--------------------------------------------------")
