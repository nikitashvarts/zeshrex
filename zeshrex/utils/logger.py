import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from zeshrex import PROJECT_PATH


def init_logger(
    name: Optional[str] = None, file_name: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Initializes a logger for handling the output. Includes 2 handlers:
        - StreamHandler (always active) for printing to console
        - FileHandler (optional) for printing to file

    :param name: Name of logger to use.
    :param file_name: (optional) Name of the file for printing the output. It will be placed in the
                      project defined directory with logs.
    :param level: Level of visible logging.
    :return: Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(filename)s : %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if file_name:
        dt_prefix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        random_prefix = uuid.uuid4().hex[:6]
        file_path = Path(PROJECT_PATH) / 'logs' / f'{dt_prefix}_{random_prefix}_{file_name}'
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
