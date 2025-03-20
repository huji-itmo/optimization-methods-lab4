import math
import numpy as np
from autograd import grad
from numpy import linalg as LA
from typing import Final


class optimization_results(object):
    """docstring for optimization_results."""

    def __init__(
        self, local_minimum: np.ndarray, iteration: int, path: list[np.ndarray]
    ):
        super(optimization_results, self).__init__()
        self.local_minimum = local_minimum
        self.iteration = iteration
        self.path = path


STOP_GRADIENT_VALUE: Final[float] = 0.0001
LEARNING_RATE: Final[float] = 0.05
VELOCITY_CHANGE_RATE: Final[float] = 0.9
EPSILON: Final[float] = 1e-8  # сглаживающий параметр, чтобы не делить на ноль
BETA_2: Final[float] = 0.99  # for adam


def sdg(
    starting_point: np.ndarray, differentiable_function, max_iterations: int
) -> optimization_results:
    """
    SDG - stochastic_gradient_descent
    """

    gradient = grad(differentiable_function)

    # Initial point
    weights = starting_point
    # Список для хранения пути точки
    path = [weights.copy()]
    i = 1
    for i in range(1, max_iterations + 1):
        grad_value = gradient(weights)

        # $W_{t+1} = W_{t} - \alpha \nabla W_{t}
        # W_{t} - weights (в нашем случае, просто аргументы функции)
        # \alpha - learning_rate
        # \nabla W_{t} = grad W_{t} - grad_value
        weights = weights - (LEARNING_RATE * grad_value)

        # Сохраняем текущую точку в пути
        path.append(weights.copy())
        if LA.norm(grad_value) < STOP_GRADIENT_VALUE:
            break

    return optimization_results(weights, i, path)


def sdg_with_momentum(
    starting_point: np.ndarray, differentiable_function, max_iterations: int
) -> optimization_results:
    """
    SDG - stochastic_gradient_descent
    with added momentum, descends more
    quickly when grad values are similar
    in a row.
    """
    gradient = grad(differentiable_function)

    # Initial point
    weights = starting_point

    velocity = 0
    # expressed as $\beta$ in math notation

    # Список для хранения пути точки
    path = [weights.copy()]
    i = 1
    for i in range(1, max_iterations + 1):
        grad_value = gradient(weights)
        # $V_{t+1} = \beta V_{t} + (1 - \beta) \nabla W_t$
        # W_{t} - weights (в нашем случае, просто аргументы функции)
        velocity = (
            velocity * VELOCITY_CHANGE_RATE - (1 - VELOCITY_CHANGE_RATE) * grad_value
        )

        # $W_{t+1} = W_{t} - \alpha V_t
        # \alpha - learning_rate
        weights = weights + LEARNING_RATE * velocity

        # Сохраняем текущую точку в пути
        path.append(weights.copy())
        if LA.norm(grad_value) < STOP_GRADIENT_VALUE:
            break

    return optimization_results(weights, i, path)


def nag(
    starting_point: np.ndarray, differentiable_function, max_iterations: int
) -> optimization_results:
    """
    NAG - Nesterov Accelerated Gradients
    a modified version of
    """
    gradient = grad(differentiable_function)

    # Initial point
    weights = starting_point

    velocity = 0
    # expressed as $\beta$ in math notation

    # Список для хранения пути точки
    path = [weights.copy()]
    i = 1
    for i in range(1, max_iterations + 1):
        # W_{t} - weights (в нашем случае, просто аргументы функции)

        # $\nabla W_t(W_t - \beta V_{t-1})$
        grad_value = gradient(weights + velocity * VELOCITY_CHANGE_RATE)

        # $V_{t} = \beta V_{t-1} + (1 - \beta) \nabla W_t$
        velocity = velocity * VELOCITY_CHANGE_RATE - LEARNING_RATE * grad_value

        # $W_{t+1} = W_{t} - \alpha V_t
        # \alpha - learning_rate
        weights = weights + velocity

        # Сохраняем текущую точку в пути
        path.append(weights.copy())
        if LA.norm(grad_value) < STOP_GRADIENT_VALUE:
            break

    return optimization_results(weights, i, path)


def adaptive_grad(
    starting_point: np.ndarray, differentiable_function, max_iterations: int
) -> optimization_results:
    """
    adaptive_grad - AdaGrad (adaptive gradient)
    """

    gradient = grad(differentiable_function)

    # Initial point
    weights = starting_point
    # Список для хранения пути точки
    path = [weights.copy()]
    velocity = 0

    i = 1
    for i in range(1, max_iterations + 1):
        grad_value = gradient(weights)

        # $s_{t} = s_{t-1} + \nabla W_t^2$
        velocity = velocity + np.power(grad_value, np.array([2, 2]))

        # $W_{t+1} = W_{t} - \alpha \dfrac{\nabla W_{t}}{\sqrt{s_t + \varepsilon}}$
        # W_{t} - weights (в нашем случае, просто аргументы функции)
        # \alpha - learning_rate
        # \nabla W_{t} = grad W_{t} - grad_value
        weights = weights - (LEARNING_RATE / (np.sqrt(velocity)) + EPSILON) * grad_value
        # Сохраняем текущую точку в пути
        path.append(weights.copy())
        if LA.norm(grad_value) < STOP_GRADIENT_VALUE:
            break

    return optimization_results(weights, i, path)


def RMSprop(
    starting_point: np.ndarray, differentiable_function, max_iterations: int
) -> optimization_results:
    """
    Root mean squared propagation
    """

    gradient = grad(differentiable_function)

    # Initial point
    weights = starting_point
    # Список для хранения пути точки
    path = [weights.copy()]
    velocity = 0

    i = 1
    for i in range(1, max_iterations + 1):
        grad_value = gradient(weights)

        # $s_{t} = s_{t-1} + \nabla W_t^2$
        velocity = velocity * VELOCITY_CHANGE_RATE + (
            1 - VELOCITY_CHANGE_RATE
        ) * np.power(grad_value, np.array([2, 2]))

        # $W_{t+1} = W_{t} - \alpha \dfrac{\nabla W_{t}}{\sqrt{s_t + \varepsilon}}$
        # W_{t} - weights (в нашем случае, просто аргументы функции)
        # \alpha - learning_rate
        # \nabla W_{t} = grad W_{t} - grad_value
        weights = weights - (LEARNING_RATE / (np.sqrt(velocity)) + EPSILON) * grad_value
        # Сохраняем текущую точку в пути
        # print(np.sqrt(velocity))
        path.append(weights.copy())
        if LA.norm(velocity) < 0.5:
            break

    return optimization_results(weights, i, path)


def adam(
    starting_point: np.ndarray, differentiable_function, max_iterations: int
) -> optimization_results:
    """
    Adam - adaptive moments
    """

    gradient = grad(differentiable_function)

    # Initial point
    weights = starting_point
    # Список для хранения пути точки
    path = [weights.copy()]
    correction = 0
    velocity = 0
    i = 1
    for i in range(1, max_iterations + 1):
        grad_value = gradient(weights)
        # $M_t$
        velocity = (
            velocity * VELOCITY_CHANGE_RATE + (1 - VELOCITY_CHANGE_RATE) * grad_value
        )

        # $V_t$
        correction = correction * BETA_2 + (1 - BETA_2) * np.power(
            grad_value, np.array([2, 2])
        )

        # we need to correct them on smaller values

        correction_hat = correction / (1 - BETA_2**i)
        velocity_hat = correction / (1 - VELOCITY_CHANGE_RATE**i)

        # $W_{t+1} = W_{t} - \dfrac{\alpha}{\sqrt{V_t} + \varepsilon} \cdot M_t$
        # W_{t} - weights (в нашем случае, просто аргументы функции)
        # \alpha - learning_rate
        # \nabla W_{t} = grad W_{t} - grad_value
        weights = weights - (
            (LEARNING_RATE * velocity_hat) / (np.sqrt(correction_hat)) + EPSILON
        )
        # Сохраняем текущую точку в пути

        path.append(weights.copy())
        if LA.norm(correction) < 0.5:
            break

    return optimization_results(weights, i, path)
