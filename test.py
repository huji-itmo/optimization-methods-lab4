import time
import numpy as np

from optimizations import optimization_results
from plot import plot_function, plot_path, plot_save, plot_show

def differentiable_function(xy):
    x, y = xy
    return x**3 - 12 * x * y + 8 * y**3


def test_optimization_method(optimization_method_func, method_name: str) -> dict[str, str]:
    start_time = time.time()
    initial_point = np.array([3.0, 1.0])
    results: optimization_results = optimization_method_func(initial_point, differentiable_function, 500)

    # Преобразуем список в массив NumPy
    plot_function(differentiable_function)
    plot_path(np.array(results.path))
    plot_save(f"output/{method_name}_path.png")
    # plot_show()

    decimals_round = 6

    results_dictionary = dict[str, str]()
    results_dictionary[f"Method name"] = method_name
    results_dictionary[f"Local minimum (x, y)"] = f"{results.local_minimum.round(decimals_round)}"
    results_dictionary[f"Function value"] = f"{round(differentiable_function(results.local_minimum), decimals_round)}"
    results_dictionary[f"Iteration count"] = f"{results.iteration}"
    results_dictionary[f"Time (seconds)"] = f"{round(time.time() - start_time, decimals_round)}"

    return results_dictionary
