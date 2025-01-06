import numpy as np
from autograd import grad
from numpy import linalg as LA

class optimization_results(object):
    """docstring for optimization_results."""

    def __init__(self, local_minimum: np.ndarray, iteration: int, path: list[np.ndarray]):
        super(optimization_results, self).__init__()
        self.local_minimum = local_minimum
        self.iteration = iteration
        self.path = path


def sdg(
    starting_point: np.ndarray,
    differentiable_function,
    max_iterations: int
) -> optimization_results:
    """
    SDG - stochastic_gradient_descent
    """
    learning_rate = 0.005
    stop_gradient_value = 0.01

    gradient = grad(differentiable_function)

    # Initial point
    weights = starting_point
    # Список для хранения пути точки
    path = [weights.copy()]
    i = 1;
    for i in range(1, max_iterations + 1):
        grad_value = gradient(weights)

        # $W_{t+1} = W_{t} - \alpha \nabla W_{t}
        # W_{t} - weights (в нашем случае, просто аргументы функции)
        # \alpha - learning_rate
        # \nabla W_{t} = grad W_{t} - grad_value
        weights = weights - (learning_rate * grad_value)

        # Сохраняем текущую точку в пути
        path.append(weights.copy())
        if LA.norm(grad_value) < stop_gradient_value:
            break;

    return optimization_results(weights, i, path)
