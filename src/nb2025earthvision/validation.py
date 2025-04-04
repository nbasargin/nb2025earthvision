import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import nb2025earthvision as ev25
from nb2025earthvision import constants, datasets


def get_moisture_rmse_on_dataset(model: ev25.MoisturePredictor, dataset: datasets.LabeledDataset):
    t3s, incs, target = dataset.t3_tensors, dataset.inc_tensors, dataset.sm_tensors
    prediction = model.predict_soil_moisture(t3s, incs)
    return ev25.get_rmse(prediction, target)


def _add_axes(fig: plt.Figure, left_inch, bottom_inch, width_inch, height_inch) -> plt.Axes:
    fig_w, fig_h = fig.get_size_inches()
    axis_box = [left_inch / fig_w, bottom_inch / fig_h, width_inch / fig_w, height_inch / fig_h]
    ax = fig.add_axes(axis_box)
    return ax


def evaluate_model_on_supervised_datasets(
    model: ev25.MoisturePredictor,
    dataset_dict: dict[str, datasets.LabeledDataset],
    title: str,
    save_to: Path,
):
    # figure setup
    num_cols = len(dataset_dict)
    fig_width = 12
    padding_top, padding_right, padding_bottom, padding_left = 0.5, 0.2, 0.5, 0.6
    gap = 0.4
    cell_size = (fig_width - gap * (num_cols - 1) - padding_left - padding_right) / num_cols
    fig_height = padding_top + cell_size + padding_bottom
    fig = plt.figure(figsize=(fig_width, fig_height))
    axs = []
    for col in range(num_cols):
        left_inch = padding_left + (cell_size + gap) * col
        axs.append(_add_axes(fig, left_inch, padding_bottom, cell_size, cell_size))
    # validation
    for i, (ds_title, dataset) in enumerate(dataset_dict.items()):
        t3s, incs, target_all = dataset.t3_tensors, dataset.inc_tensors, dataset.sm_tensors
        prediction_all = model.predict_soil_moisture(t3s, incs)
        metrics_str = f"ALL {len(prediction_all)} points\n"
        metrics_str += f"RMSE = {ev25.get_rmse(prediction_all, target_all) * 100:.2f} vol.%\n"
        metrics_str += f"Bias = {ev25.get_bias(prediction_all, target_all) * 100:.2f} vol.%\n"
        metrics_str += f"Pearson = {ev25.pearson_corrcoef(prediction_all * 100, target_all * 100):.2f}"
        # valid only
        valid_points = (prediction_all > constants.soil_mst_min) & (prediction_all < constants.soil_mst_max)
        prediction_valid = prediction_all[valid_points]
        target_valid = target_all[valid_points]
        metrics_valid_str = f"VALID {len(prediction_valid)} points\n"
        metrics_valid_str += f"RMSE = {ev25.get_rmse(prediction_valid, target_valid) * 100:.2f} vol.%\n"
        metrics_valid_str += f"Bias = {ev25.get_bias(prediction_valid, target_valid) * 100:.2f} vol.%\n"
        metrics_valid_str += f"Pearson = {ev25.pearson_corrcoef(prediction_valid * 100, target_valid * 100):.2f}"
        # plot
        ax = axs[i]
        ax.scatter(x=target_all.cpu().detach().numpy() * 100, y=prediction_all.cpu().detach().numpy() * 100, c="red")
        ax.scatter(
            x=target_valid.cpu().detach().numpy() * 100, y=prediction_valid.cpu().detach().numpy() * 100, c="blue"
        )
        ax.text(0.04, 0.96, metrics_str, transform=ax.transAxes, verticalalignment="top", fontsize=7)
        ax.text(
            0.96,
            0.96,
            metrics_valid_str,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=7,
        )
        sm_min, sm_max = 0, 60
        center_line = np.array([sm_min, sm_max])
        ax.plot(center_line, center_line, color="black", linestyle="dashed", alpha=0.2)  # center line
        ax.plot(center_line, center_line + 10, color="black", linestyle="dotted", alpha=0.1)  # +10 line
        ax.plot(center_line, center_line - 10, color="black", linestyle="dotted", alpha=0.1)  # -10 line
        if i == 0:
            ax.set_ylabel("predicted moisture in vol.%")
        else:
            ax.set_yticks([])
        ax.set_xlabel("ground truth moisture in vol.%")
        ax.set_xlim(sm_min, sm_max)
        ax.set_ylim(sm_min, sm_max)
        ax.set_title(ds_title)
    fig.suptitle(title)
    fig.savefig(save_to, dpi=300)
    plt.close("all")
