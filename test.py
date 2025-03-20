import time
import numpy as np

from optimizations import optimization_results
from plot import plot_function, plot_path, plot_save, plot_show


def differentiable_function(xy):
    x, y = xy
    return x**2 + 3 * y**2 + 3 * x * y - x - 2 * y - 1


def test_optimization_method(
    optimization_method_func, method_name: str
) -> dict[str, str]:
    start_time = time.time()
    initial_point = np.array([3.0, 3.0])
    results: optimization_results = optimization_method_func(
        initial_point, differentiable_function, 1000
    )
    end_time = time.time()
    # Преобразуем список в массив NumPy
    plot_function(differentiable_function)
    plot_path(np.array(results.path), method_name)
    plot_save(f"output/{method_name}_path.png")
    # plot_show()

    decimals_round = 4

    results_dictionary = dict[str, str]()
    results_dictionary[f"Method name"] = method_name
    results_dictionary[f"Local minimum (x, y)"] = (
        f"{results.local_minimum.round(decimals_round)}"
    )
    results_dictionary[f"Function value"] = (
        f"{round(differentiable_function(results.local_minimum), decimals_round)}"
    )
    results_dictionary[f"Iteration count"] = f"{results.iteration}"
    results_dictionary[f"Time (seconds)"] = (
        f"{round(end_time - start_time, decimals_round)}"
    )

    return results_dictionary
