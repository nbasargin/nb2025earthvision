import matplotlib.pyplot as plt
import shapely
from shapely.plotting import plot_polygon


def plot_polygon_into_axis(ax, polygon, flip_xy=False, facecolor="none", edgecolor="black", linewidth=2):
    if flip_xy:
        polygon = shapely.ops.transform(lambda x, y: (y, x), polygon)
    plot_polygon(polygon, ax=ax, facecolor=facecolor, add_points=False, edgecolor=edgecolor, linewidth=linewidth)


def plot_parameter_history(param_hist_dict, title, save_to, xlabel=None, ylabel=None, ylim=(None, None), alpha=1):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(left=0.17, bottom=0.15, right=0.92, top=0.88)
    for key, param_hist in param_hist_dict.items():
        ax.plot(param_hist, label=key, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_to, dpi=300)
    plt.close("all")
