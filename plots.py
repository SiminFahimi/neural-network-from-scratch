from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PlotCurve:
    data: List[Tuple[list, list]]
    x_label: str
    y_label: str
    line_label: list[str]
    title: str
    save_address: str

def plot_curves(curve: PlotCurve):
    plt.figure()
    for (x_vals, y_vals), label in zip(curve.data, curve.line_label):
        plt.plot(x_vals, y_vals, label=label)

    plt.xlabel(curve.x_label)
    plt.ylabel(curve.y_label)
    plt.title(curve.title)
    plt.legend()
    os.makedirs(os.path.dirname(curve.save_address) or ".", exist_ok=True)
    plt.savefig(curve.save_address)
    plt.show()

def plot_3d_predictions(x_data, y_true, y_pred, x_std, x_mean, surface_func):


    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    correct = (y_pred == y_true)
    misclassified = (y_pred != y_true).flatten()

    correct_positive = (correct & (y_pred == 1)).flatten()
    correct_negative = (correct & (y_pred == 0)).flatten()

    print("correct_negative count:", correct_negative.sum())
    print("correct_positive count:", correct_positive.sum())
    print("misclassified count:", misclassified.sum())

    x_data = x_data * x_std + x_mean

    x_vals = np.linspace(-10, 10, 60)
    y_vals = np.linspace(-10, 10, 60)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    z_grid = surface_func(x_grid, y_grid)

    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.4)

    ax.scatter(
        x_data[correct_positive, 0],
        x_data[correct_positive, 1],
        x_data[correct_positive, 2],
        s=10,
        c="green",
        label="Correct Positive",
    )

    ax.scatter(
        x_data[correct_negative, 0],
        x_data[correct_negative, 1],
        x_data[correct_negative, 2],
        s=10,
        c="blue",
        label="Correct Negative",
    )

    ax.scatter(
        x_data[misclassified, 0],
        x_data[misclassified, 1],
        x_data[misclassified, 2],
        s=12,
        c="red",
        label="Misclassified",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.title("3D Predictions + Decision Surface")

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/prediction_3d.png", dpi=300, bbox_inches="tight")
    plt.show()