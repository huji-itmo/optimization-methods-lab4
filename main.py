from latex import get_latex_table
from optimizations import *
from test import test_optimization_method


if __name__ == "__main__":
    functions_to_test = {
        "stochastic_gradient_descent": sdg,
        "AdaGrad": adaptive_grad,
        "sdg_with_momentum": sdg_with_momentum,
        "Nesterov Accelerated Gradients": nag,
        "RMSprop": RMSprop,
        "Adam": adam
    }
    all_results = list[dict[str, str]]()
    for method_name, method_func in functions_to_test.items():
        results = test_optimization_method(method_func, method_name)
        all_results.append(results)

    with open("output/final_table.tex", "w") as file:
        table_str = get_latex_table(all_results)
        file.write(table_str)
