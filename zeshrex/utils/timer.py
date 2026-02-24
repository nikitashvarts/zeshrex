from time import time
from typing import Callable, Optional


class Timer:
    def __init__(self, description: Optional[str] = None, print_function: Callable = print):
        self.description = description
        self.print_function = print_function
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time()

    def __exit__(self, *args, **kwargs):
        self.end = time()
        if self.description is not None:
            print_message = f'Time elapsed for "{self.description}": {self.end - self.start} seconds'
        else:
            print_message = f'Time elapsed: {self.end - self.start} seconds'
        self.print_function(print_message)
