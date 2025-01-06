from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA

def plot_path(path):

    for i in range(len(path)):
        if (LA.norm(path[i]) > 20):
            path = path[:i-1]
            break;

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
    x_vals = np.linspace(0, 4, 600)
    y_vals = np.linspace(-1, 5, 600)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = func(np.array([X, Y]))

    z_levels_span = 20;

    contour = plt.contour(X, Y, Z,
        levels=np.linspace(-z_levels_span, z_levels_span, z_levels_span+1),
        cmap="viridis")
    plt.colorbar(contour)

def plot_show():
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

def plot_save(path: str):
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=1000)
    plt.close()
