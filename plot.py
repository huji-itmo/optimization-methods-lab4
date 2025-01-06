

from matplotlib import pyplot as plt
import numpy as np

def plot_path(path):
    plt.plot(
        path[:, 0],
        path[:, 1],
        marker="o",
        color="violet",
        markersize=5,
        label="Path to minimum",
    )

    plt.text(path[0, 0], path[0, 1], 'Starting\n point', fontsize=8)

def plot_function(func):
    x_vals = np.linspace(-6, 6, 600)
    y_vals = np.linspace(-6, 6, 600)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = func(np.array([X, Y]))

    z_levels_span = 20;

    contour = plt.contour(X, Y, Z,
        levels=np.linspace(-z_levels_span, z_levels_span, z_levels_span+1),
        cmap="viridis")
    plt.colorbar(contour)

def plot_show():
    plt.title("Path to Local Minimum of the Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

def plot_save(path: str):
    plt.title("Path to Local Minimum of the Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.savefig(path)
