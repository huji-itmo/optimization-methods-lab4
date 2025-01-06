import time
import autograd.numpy as np
from autograd import grad
from numpy import linalg as LA

from plot import plot_function, plot_path, plot_show


def differentiable_function(xy):
    x, y = xy
    return x**3 - 12 * x * y + 8 * y**3


start_time = time.time()
gradient = grad(differentiable_function)

# Initial point
initial_point = np.array([5.0, 1.0])
arg = initial_point

learning_rate = 0.005
stop_criteria_value = 0.01
# Список для хранения пути точки
path = [arg.copy()]

for i in range(999):
    grad_value = gradient(arg)

    #(x_{k+1}, y_{k+1}) = (x_{k}, y_{k}) - a_k * grad f(x_{k}, y_{k})
    arg = arg - (learning_rate * grad_value)
    # Сохраняем текущую точку в пути
    path.append(arg.copy())

    print(f"Iteration {i + 1}:")
    print("Gradient: ", grad_value)
    print("Updated point: ", arg)
    # print("Norm of gradient: ", LA.norm(grad_value))

    if LA.norm(grad_value) < stop_criteria_value:
        end_time = time.time()
        print(f"\n\n")
        print(f"Local minimum found at (x, y): {arg}")
        print(f"Function value: {differentiable_function(arg)}")
        print(f"Iteration count: {i+1}")
        print(f"Learning value: a_k = {learning_rate}")
        print(f"Stop criteria: || grad f(x_k, y_k) || < {stop_criteria_value}")
        print(f"Time: {end_time - start_time:.4f} s")
        break


# Преобразуем список в массив NumPy
path = np.array(path)

plot_path(path)
plot_function(differentiable_function)
plot_show()
