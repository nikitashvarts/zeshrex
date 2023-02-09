import os
from pathlib import Path

from zeshrex.utils.config_loader import load_yaml_config, print_configs

PROJECT_PATH = Path(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')))
TMP_DIRNAME = PROJECT_PATH / 'tmp'

CONFIG_FILE_PATH = PROJECT_PATH / 'config' / 'config.yaml'
