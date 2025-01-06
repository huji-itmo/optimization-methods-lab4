import numpy as np


class optimization_results(object):
    """docstring for optimization_results."""

    def __init__(self, local_minimum: np.ndarray, iteration: int, path: list[np.ndarray]):
        super(optimization_results, self).__init__()
        self.local_minimum = local_minimum
        self.iteration = iteration
        self.path = path
