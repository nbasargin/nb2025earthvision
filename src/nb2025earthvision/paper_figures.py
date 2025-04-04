import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
import fsarcamp as fc

import nb2025earthvision as ev25
from nb2025earthvision import datasets, constants, plot_functions, campaign_data

# internal constants
_WIDTH_1COL = 3 + 1 / 4  # single column width
_WIDTH_2COL = 6 + 7 / 8  # full width


def setup_matplotlib_paper_figure_styles():
    """Setup global matplotlib styles for paper figures."""
    # Use Times New Roman if provided
    local_times_new_roman_font = ev25.get_root_folder() / "times.ttf"
    if local_times_new_roman_font.exists():
        font_manager.fontManager.addfont(local_times_new_roman_font)
        print("Using Times New Roman ttf")
    else:
        print("Using default fonts, figures might look off if Times New Roman is not installed")
    # Other params
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["font.size"] = 6
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["axes.titlesize"] = 6
    plt.rcParams["axes.labelsize"] = 6
    plt.rcParams["axes.titlepad"] = 3
    plt.rcParams["axes.labelpad"] = 2
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["figure.dpi"] = 450


def get_selected_regions_and_passes():
    """Selected F-SAR passes and regions for comparison plots."""
    label_pass_regions = [
        ("CR maize", "14cropex0914", [ev25.CORN_C2_TRAIN, ev25.CORN_C2_VAL, ev25.CORN_C1]),
        ("HT wheat", "22hterra0404", [ev25.CREA_DW]),
        ("HT bare soil", "22hterra0104", [ev25.CREA_BS_QU]),
        ("HT maize", "22hterra0504", [ev25.CAIONE_MA]),
    ]
    return label_pass_regions


def _get_color(i):
    return list(mcolors.TABLEAU_COLORS.values())[i]


def _add_axes(fig: plt.Figure, left_inch, bottom_inch, width_inch, height_inch) -> plt.Axes:
    fig_w, fig_h = fig.get_size_inches()
    axis_box = [left_inch / fig_w, bottom_inch / fig_h, width_inch / fig_w, height_inch / fig_h]
    ax = fig.add_axes(axis_box)
    return ax


def _setup_scatter_ax(ax: plt.Axes):
    ax.yaxis.tick_right()
    sm_min, sm_max, sm_top_buffer = 0, 50, 25
    center_line = np.array([sm_min, sm_max])
    ax.plot(center_line, center_line, color="black", linestyle="dashed", linewidth=0.5, alpha=0.2)  # center line
    ax.plot(center_line, center_line + 10, color="black", linestyle="dotted", linewidth=0.5, alpha=0.1)  # +10 line
    ax.plot(center_line, center_line - 10, color="black", linestyle="dotted", linewidth=0.5, alpha=0.1)  # -10 line
    ax.set_xlim(sm_min, sm_max)
    ax.set_ylim(sm_min, sm_max + sm_top_buffer)
    ax.set_xticks([5, 25, 45])
    ax.set_yticks([5, 25, 45])


def _plot_scatters_into_axis(ax: plt.Axes, models: list[ev25.MoisturePredictor], dataset: datasets.LabeledDataset):
    rmse_list, bias_list, pearson_list = [], [], []
    for idx, model in enumerate(models):
        color = _get_color(idx)
        predicted = model.predict_soil_moisture(dataset.t3_tensors, dataset.inc_tensors)
        target = dataset.sm_tensors
        ax.scatter(
            x=target.cpu().detach().numpy() * 100,
            y=predicted.cpu().detach().numpy() * 100,
            c=color,
            s=4,
            linewidth=0,
            alpha=0.2,
        )
        rmse = ev25.get_rmse(predicted, target)
        bias = ev25.get_bias(predicted, target).item()
        pearson = ev25.pearson_corrcoef(predicted * 100, target * 100).item()
        rmse_list.append(rmse)
        bias_list.append(bias)
        pearson_list.append(pearson)
    rmse_mean = np.mean(rmse_list) * 100
    rmse_std = np.std(rmse_list) * 100
    bias_mean = np.mean(bias_list) * 100
    bias_std = np.std(bias_list) * 100
    pearson_mean = np.mean(pearson_list)
    pearson_std = np.std(pearson_list)
    metrics_str = f"RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}\n"
    metrics_str += f"Bias: {bias_mean:.2f} ± {bias_std:.2f}\n"
    metrics_str += f"Pearson: {pearson_mean:.2f} ± {pearson_std:.2f}"
    ax.text(0.03, 0.97, metrics_str, transform=ax.transAxes, verticalalignment="top", fontsize=5)


def _scatter_grid_layout(num_rows, num_cols, fig_width=_WIDTH_1COL):
    padding_top, padding_right, padding_bottom, padding_left = 0.12, 0.28, 0.27, 0.12
    gap = 0.07
    cell_w = (fig_width - gap * (num_cols - 1) - padding_left - padding_right) / num_cols
    cell_h = cell_w * 1.1  # cell_w / (sm_max + sm_min) * (sm_max + sm_min + sm_top_buffer)
    fig_height = padding_top + cell_h * num_rows + gap * (num_rows - 1) + padding_bottom
    fig = plt.figure(figsize=(fig_width, fig_height))
    axs = []
    # axis grid
    for row in range(num_rows):
        bottom_inch = padding_bottom + (cell_h + gap) * (num_rows - row - 1)
        row_axs = []
        for col in range(num_cols):
            left_inch = padding_left + (cell_w + gap) * col
            row_axs.append(_add_axes(fig, left_inch, bottom_inch, cell_w, cell_h))
        axs.append(row_axs)
    return fig, np.array(axs)


def _plot_scatter_grid(ds_list: list[datasets.LabeledDataset], seeds):
    look_mode = constants.look_mode
    version = constants.code_version
    m_s, plant_mst, phi = constants.m_s, constants.plant_mst, constants.phi
    sand, clay, frequency = constants.sand, constants.clay, constants.frequency
    # models
    supervised_models, unsupervised_models, hybrid_models, physical_models = [], [], [], []
    for seed in seeds:
        supervised_path = ev25.get_supervised_model_path(look_mode, version, seed)
        unsupervised_path = ev25.get_selfsupervised_model_path(look_mode, version, seed)
        hybrid_path = ev25.get_hybrid_model_path(look_mode, version, seed)
        physical_model = ev25.PhysicalInversionModel(
            m_s=m_s, plant_mst=plant_mst, phi=phi, sand=sand, clay=clay, frequency=frequency, seed=seed
        )
        supervised_models.append(ev25.load_encoder(supervised_path))
        unsupervised_models.append(ev25.load_encoder(unsupervised_path))
        hybrid_models.append(ev25.load_encoder(hybrid_path))
        physical_models.append(physical_model)
    model_dict = {
        "Supervised": supervised_models,
        "Hybrid AE": hybrid_models,
        "Self-sup. AE": unsupervised_models,
        "Physical": physical_models,
    }
    # figure setup
    num_rows, num_cols = len(ds_list), 4
    fig, axs = _scatter_grid_layout(num_rows=num_rows, num_cols=num_cols)
    for row, dataset in enumerate(ds_list):
        for col, (model_label, models) in enumerate(model_dict.items()):
            ax = axs[row, col]
            _setup_scatter_ax(ax)
            _plot_scatters_into_axis(ax, models, dataset)
            # titles and labels
            if row == 0:
                ax.set_title(model_label)
                ax.set_xticklabels([])
            elif row == len(axs) - 1:
                ax.set_xlabel("measured $w_s$", labelpad=0)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(dataset.identifier)
                ax.set_yticklabels([])
            elif col == num_cols - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel("predicted $w_s'$", labelpad=0)
            else:
                ax.set_yticklabels([])
    return fig


def _plot_pauli_preview_into_axis(pass_name, band, region_names, ax):
    az_min, az_max, rg_min, rg_max = campaign_data.get_region_slc_extent(pass_name, band, region_names)
    data_extent = (rg_min, rg_max, az_min, az_max)  # (left, right, bottom, top) in data coordinates
    slc_hh = campaign_data.get_region_slc(pass_name, band, "hh", az_min, az_max, rg_min, rg_max)
    slc_hv = campaign_data.get_region_slc(pass_name, band, "hv", az_min, az_max, rg_min, rg_max)
    slc_vh = campaign_data.get_region_slc(pass_name, band, "vh", az_min, az_max, rg_min, rg_max)
    slc_vv = campaign_data.get_region_slc(pass_name, band, "vv", az_min, az_max, rg_min, rg_max)
    window_size = campaign_data.get_window_size(pass_name, band, "preview")
    pauli_rgb_max = campaign_data.get_pauli_rgb_max(band)
    rgb = np.stack(fc.slc_to_pauli_rgb(slc_hh, slc_hv, slc_vh, slc_vv, *window_size, pauli_rgb_max), axis=2)
    rgb[np.isnan(rgb)] = 0
    ax.imshow(rgb, extent=data_extent, origin="lower", aspect="auto")


def _fig_labeled_dataset_layout() -> tuple[plt.Figure, np.ndarray]:
    """CROPEX time series preview, 2x7 grid, no colorbars"""
    num_cols = 7
    num_rows = 2
    fig_width = _WIDTH_2COL
    padding_top, padding_right, padding_bottom, padding_left = 0.16, 0.16, 0.01, 0.16
    gap = 0.03
    cell_size = (fig_width - gap * (num_cols - 1) - padding_left - padding_right) / num_cols
    fig_height = padding_top + cell_size * num_rows + gap * (num_rows - 1) + padding_bottom
    fig = plt.figure(figsize=(fig_width, fig_height))
    axs = []
    for row in range(num_rows):
        bottom_inch = padding_bottom + (cell_size + gap) * (num_rows - row - 1)
        row_axs = []
        for col in range(num_cols):
            left_inch = padding_left + (cell_size + gap) * col
            row_axs.append(_add_axes(fig, left_inch, bottom_inch, cell_size, cell_size))
        axs.append(row_axs)
    return fig, np.array(axs)


def fig_labeled_dataset():
    band = constants.band
    version = constants.code_version
    pass_names = [
        "14cropex0210",
        "14cropex0305",
        "14cropex0620",
        "14cropex0718",
        "14cropex0914",
        "14cropex1114",
        "14cropex1318",
    ]
    region_names = [ev25.CORN_C2_TRAIN, ev25.CORN_C2_VAL, ev25.CORN_C1]
    regions = ev25.EarthVision2025Regions()
    pauli_rgb_max = campaign_data.get_pauli_rgb_max(band)
    fig, axs = _fig_labeled_dataset_layout()
    for i, pass_name in enumerate(pass_names):
        # optical image
        ax_optical = axs[0, i]
        ax_optical.set_title(campaign_data.pass_name_to_date_label(pass_name), fontsize=8)
        ax_optical.imshow(campaign_data.get_cropex_maize_image_by_pass(pass_name))
        ax_optical.set_xticks([])
        ax_optical.set_yticks([])
        # pauli image
        ax_pauli = axs[1, i]
        az_min, az_max, rg_min, rg_max = campaign_data.get_region_slc_extent(pass_name, band, region_names)
        data_extent = (rg_min, rg_max, az_min, az_max)  # (left, right, bottom, top) in data coordinates
        window_size = campaign_data.get_window_size(pass_name, band, "preview")
        region_t3 = campaign_data.get_region_t3(pass_name, band, az_min, az_max, rg_min, rg_max, window_size)
        rgb = np.stack(fc.coherency_matrix_to_pauli_rgb(region_t3, pauli_rgb_max), axis=2)
        ax_pauli.imshow(rgb, extent=data_extent, origin="lower", aspect="auto")
        ax_pauli.set_xticks([])
        ax_pauli.set_yticks([])
        # polygons
        campaign = campaign_data.get_campaign(pass_name)
        lut = campaign.get_pass(pass_name, band).load_gtc_sr2geo_lut()
        for region_i, region_name in enumerate(region_names):
            color = _get_color(region_i)
            geometry = regions.get_geometry_azrg(region_name, lut)
            plot_functions.plot_polygon_into_axis(ax_pauli, geometry, flip_xy=True, edgecolor="black", linewidth=2)
            plot_functions.plot_polygon_into_axis(ax_pauli, geometry, flip_xy=True, edgecolor=color, linewidth=1.5)
        if i == 0:
            ax_optical.set_ylabel("Field conditions", fontsize=8)
            ax_pauli.set_ylabel("Pauli RGB", fontsize=8)
    fig.savefig(ev25.get_paper_figures_folder() / f"fig_labeled_dataset_v{version}.jpg")
    plt.close("all")


def _fig_fsar_pauli_slc_layout(num_rows) -> tuple[plt.Figure, np.ndarray]:
    """SLC images over each other"""
    fig_width = _WIDTH_1COL
    padding_top, padding_right, padding_bottom, padding_left = 0.01, 0.01, 0.01, 0.12
    gap = 0.03
    cell_width = fig_width - padding_left - padding_right
    cell_height = 0.8
    fig_height = padding_top + cell_height * num_rows + gap * (num_rows - 1) + padding_bottom
    fig = plt.figure(figsize=(fig_width, fig_height))
    axs = []
    for row in range(num_rows):
        bottom_inch = padding_bottom + (cell_height + gap) * (num_rows - row - 1)
        axs.append(_add_axes(fig, padding_left, bottom_inch, cell_width, cell_height))
    return fig, np.array(axs)


def fig_fsar_pauli_slc():
    band = constants.band
    version = constants.code_version
    pass_labels_names = [("CROPEX, July 2014", "14cropex1318"), ("HTERRA, June 2022", "22hterra0504")]
    pauli_rgb_max = campaign_data.get_pauli_rgb_max(band)
    fig, axs = _fig_fsar_pauli_slc_layout(len(pass_labels_names))
    for i, ax in enumerate(axs):
        label, pass_name = pass_labels_names[i]
        slc_hh = campaign_data.get_slc(pass_name, band, "hh")
        slc_hv = campaign_data.get_slc(pass_name, band, "hv")
        slc_vh = campaign_data.get_slc(pass_name, band, "vh")
        slc_vv = campaign_data.get_slc(pass_name, band, "vv")
        window_size = campaign_data.get_window_size(pass_name, band, "preview")
        rgb = np.stack(fc.slc_to_pauli_rgb(slc_hh, slc_hv, slc_vh, slc_vv, *window_size, pauli_rgb_max), axis=2)
        rgb[np.isnan(rgb)] = 0
        ax.imshow(np.rot90(rgb), origin="lower", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f"{label}")
    fig.savefig(ev25.get_paper_figures_folder() / f"fig_fsar_pauli_slc_v{version}.jpg")
    plt.close("all")


def _grid_layout_bottom_cb(
    num_rows, num_cols, default_width=_WIDTH_1COL, default_cols=5
) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    """
    R rows, C columns, horizontal colorbards below each column.
    By default the figure spans half of the page and has 5 columns.
    If less columns are provided, the width is reduced to keep the cell size.
    """
    padding_top, padding_right, padding_bottom, padding_left = 0.12, 0.01, 0.17, 0.12
    cbar_height = 0.05
    gap = 0.03
    cell_w = (default_width - gap * (default_cols - 1) - padding_left - padding_right) / default_cols
    cell_h = cell_w * 0.89 # squeeze a bit
    fig_width = cell_w * num_cols + padding_left + padding_right + gap * (num_cols - 1)
    fig_height = padding_top + cell_h * num_rows + gap * num_rows + cbar_height + padding_bottom
    fig = plt.figure(figsize=(fig_width, fig_height))
    axs = []
    cell_w_gap = cell_w + gap
    # axis grid
    for row in range(num_rows):
        bottom_inch = padding_bottom + (cell_h + gap) * (num_rows - row - 1) + cbar_height + gap
        row_axs = []
        for col in range(num_cols):
            left_inch = padding_left + (cell_w + gap) * col
            row_axs.append(_add_axes(fig, left_inch, bottom_inch, cell_w, cell_h))
        axs.append(row_axs)
    # bottom colorbars
    caxs = []
    for col in range(num_cols):
        left_inch = padding_left + cell_w_gap * col
        cax = _add_axes(fig, left_inch, padding_bottom, cell_w, cbar_height)
        caxs.append(cax)
    return fig, np.array(axs), np.array(caxs)


def _load_downsampled_data(pass_name, band, region_names, look_mode, goal_px_size):
    """Load and downsample data in the specified region."""
    az_min, az_max, rg_min, rg_max = campaign_data.get_region_slc_extent(pass_name, band, region_names)
    data_extent = (rg_min, rg_max, az_min, az_max)  # (left, right, bottom, top) in data coordinates
    window_size = campaign_data.get_window_size(pass_name, band, look_mode)
    t3 = campaign_data.get_region_t3(pass_name, band, az_min, az_max, rg_min, rg_max, window_size)
    t3 = campaign_data.mask_strong_t3_pixels(t3, constants.MAX_T3_POWER)
    incidence = campaign_data.get_region_incidence(pass_name, band, az_min, az_max, rg_min, rg_max)
    # downsampling for faster figure generation
    down_az = max(1, int((az_max - az_min) / goal_px_size))
    down_rg = max(1, int((rg_max - rg_min) / goal_px_size))
    t3, incidence = t3[::down_az, ::down_rg], incidence[::down_az, ::down_rg]
    t3, incidence = torch.tensor(t3), torch.tensor(incidence)
    return t3, incidence, data_extent


def fig_moisture_comparison():
    look_mode = constants.look_mode
    seed = 0
    version = constants.code_version
    band = constants.band
    m_s, plant_mst, phi = constants.m_s, constants.plant_mst, constants.phi
    sand, clay, frequency = constants.sand, constants.clay, constants.frequency
    goal_px_size = 300
    label_pass_regions = get_selected_regions_and_passes()
    regions = ev25.EarthVision2025Regions()
    fig, axs, caxs = _grid_layout_bottom_cb(num_rows=len(label_pass_regions), num_cols=5)
    moisture_style = dict(vmin=0, vmax=50, cmap="viridis_r", origin="lower", aspect="auto")
    for row, (label, pass_name, region_names) in enumerate(label_pass_regions):
        campaign = campaign_data.get_campaign(pass_name)
        lut = campaign.get_pass(pass_name, band).load_gtc_sr2geo_lut()
        ax_pauli, ax_sup, ax_hyb, ax_self, ax_phys = axs[row]
        t3, incidence, data_extent = _load_downsampled_data(pass_name, band, region_names, look_mode, goal_px_size)
        # pauli preview
        _plot_pauli_preview_into_axis(pass_name, band, region_names, ax_pauli)
        ax_pauli.set_ylabel(f"{label}")
        # supervised
        supervised_path = ev25.get_supervised_model_path(look_mode, version, seed)
        supervised_model = ev25.load_encoder(supervised_path)
        supervised_mst = supervised_model.predict_soil_moisture(t3, incidence)
        supervised_mst = supervised_mst.cpu().detach().numpy() * 100
        im = ax_sup.imshow(supervised_mst, extent=data_extent, **moisture_style)
        # hybrid
        hybrid_path = ev25.get_hybrid_model_path(look_mode, version, seed)
        hybrid_model = ev25.load_encoder(hybrid_path)
        hybrid_mst = hybrid_model.predict_soil_moisture(t3, incidence)
        hybrid_mst = hybrid_mst.cpu().detach().numpy() * 100
        im = ax_hyb.imshow(hybrid_mst, extent=data_extent, **moisture_style)
        # self-supervised autoencoder
        unsupervised_path = ev25.get_selfsupervised_model_path(look_mode, version, seed)
        unsupervised_model = ev25.load_encoder(unsupervised_path)
        unsupervised_mst = unsupervised_model.predict_soil_moisture(t3, incidence)
        unsupervised_mst = unsupervised_mst.cpu().detach().numpy() * 100
        im = ax_self.imshow(unsupervised_mst, extent=data_extent, **moisture_style)
        # physical model
        physical_model = ev25.PhysicalInversionModel(
            m_s=m_s, plant_mst=plant_mst, phi=phi, sand=sand, clay=clay, frequency=frequency, seed=seed
        )
        physical_mst = physical_model.predict_soil_moisture(t3, incidence)
        physical_mst = physical_mst.cpu().detach().numpy() * 100
        im = ax_phys.imshow(physical_mst, extent=data_extent, **moisture_style)
        # polygons
        for region_name in region_names:
            geometry = regions.get_geometry_azrg(region_name, lut)
            for ax in [ax_pauli, ax_sup, ax_hyb, ax_self, ax_phys]:
                plot_functions.plot_polygon_into_axis(ax, geometry, flip_xy=True, edgecolor="black", linewidth=0.5)
        # no ticks
        for ax in [ax_pauli, ax_sup, ax_hyb, ax_self, ax_phys]:
            ax.set_xticks([])
            ax.set_yticks([])
        if row == 0:
            ax_pauli.set_title("Pauli RGB")
            ax_sup.set_title("Supervised")
            ax_hyb.set_title("Hybrid AE")
            ax_self.set_title("Self-sup. AE")
            ax_phys.set_title("Physical")
    for i, cax in enumerate(caxs):
        if i > 0:
            fig.colorbar(im, cax=cax, orientation="horizontal")
            cax.set_xticks([5, 25, 45])
        else:
            cax.axis("off")  # no colorbar under pauli
    caxs[1].set_ylabel(
        "soil moisture\nin %",
        rotation=0,
        horizontalalignment="right",
        verticalalignment="top",
    )
    fig.savefig(ev25.get_paper_figures_folder() / f"fig_moisture_comparison_regions_v{version}s{seed}.png")
    plt.close("all")


def fig_train_val_test_scatter():
    band = constants.band
    look_mode = constants.look_mode
    version = constants.code_version
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # datasets
    ds_id_label = [
        (datasets.CROPEX_MA_TRAIN, "Train, $\mathcal{D}_{train}$"),
        (datasets.CROPEX_MA_VAL, "Validation, $\mathcal{D}_{val}$"),
        (datasets.CROPEX_MA_TEST, "Test, $\mathcal{D}_{test}$"),
    ]
    ds_list = []
    dataset_folder = ev25.get_dataset_folder()
    for ds_id, label in ds_id_label:
        ds = datasets.get_labeled_dataset_by_id(ds_id, band, look_mode, dataset_folder)
        ds.identifier = label
        ds_list.append(ds)
    fig = _plot_scatter_grid(ds_list, seeds)
    fig.savefig(ev25.get_paper_figures_folder() / f"fig_train_val_test_scatter_v{version}_{len(seeds)}runs.png")
    plt.close("all")


def fig_ood_scatter():
    band = constants.band
    look_mode = constants.look_mode
    version = constants.code_version
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # datasets
    ds_id_label = [
        (datasets.HTERRA_WH_CREA, "HT wheat, $\mathcal{D}_{wh}$"),
        (datasets.HTERRA_MA_CAIONE, "HT maize, $\mathcal{D}_{ma}$"),
        (datasets.CROPEX_CU, "CR cucumbers, $\mathcal{D}_{cu}$"),
        # datasets.CROPEX_WH,
        # datasets.HTERRA_MA_CREA,
        # datasets.HTERRA_BS_QU,
        # datasets.HTERRA_AA,
        # datasets.HTERRA_SF,
        # datasets.HTERRA_WH_CAIONE,
    ]
    ds_list = []
    dataset_folder = ev25.get_dataset_folder()
    for ds_id, label in ds_id_label:
        ds = datasets.get_labeled_dataset_by_id(ds_id, band, look_mode, dataset_folder)
        ds.identifier = label
        ds_list.append(ds)
    fig = _plot_scatter_grid(ds_list, seeds)
    fig.savefig(ev25.get_paper_figures_folder() / f"fig_ood_scatter_v{version}_{len(seeds)}runs.png")
    plt.close("all")


def _get_moisture_stddev_between_models(
    models: list[ev25.MoisturePredictor], t3: torch.Tensor, incidence: torch.Tensor
):
    results = []
    for model in models:
        predicted_sm = model.predict_soil_moisture(t3, incidence)
        predicted_sm = predicted_sm.cpu().detach().numpy()
        results.append(predicted_sm)
    predicted_sm_stack = np.stack(results, axis=0)
    stddev = np.std(predicted_sm_stack, axis=0)
    return stddev


def fig_stddev_comparison():
    look_mode = constants.look_mode
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    version = constants.code_version
    band = constants.band
    m_s, plant_mst, phi = constants.m_s, constants.plant_mst, constants.phi
    sand, clay, frequency = constants.sand, constants.clay, constants.frequency
    goal_px_size = 300
    label_pass_regions = get_selected_regions_and_passes()
    regions = ev25.EarthVision2025Regions()
    # load models
    supervised_models, unsupervised_models, hybrid_models, physical_models = [], [], [], []
    for seed in seeds:
        supervised_path = ev25.get_supervised_model_path(look_mode, version, seed)
        unsupervised_path = ev25.get_selfsupervised_model_path(look_mode, version, seed)
        hybrid_path = ev25.get_hybrid_model_path(look_mode, version, seed)
        physical_model = ev25.PhysicalInversionModel(
            m_s=m_s, plant_mst=plant_mst, phi=phi, sand=sand, clay=clay, frequency=frequency, seed=seed
        )
        supervised_models.append(ev25.load_encoder(supervised_path))
        unsupervised_models.append(ev25.load_encoder(unsupervised_path))
        hybrid_models.append(ev25.load_encoder(hybrid_path))
        physical_models.append(physical_model)
    fig, axs, caxs = _grid_layout_bottom_cb(num_rows=len(label_pass_regions), num_cols=5)
    stddev_style = dict(vmin=0, vmax=10, cmap="RdYlGn_r", origin="lower", aspect="auto")
    for row, (label, pass_name, region_names) in enumerate(label_pass_regions):
        campaign = campaign_data.get_campaign(pass_name)
        lut = campaign.get_pass(pass_name, band).load_gtc_sr2geo_lut()
        ax_pauli, ax_sup, ax_hyb, ax_self, ax_phys = axs[row]
        t3, incidence, data_extent = _load_downsampled_data(pass_name, band, region_names, look_mode, goal_px_size)
        # pauli preview
        _plot_pauli_preview_into_axis(pass_name, band, region_names, ax_pauli)
        ax_pauli.set_ylabel(f"{label}")
        # supervised stddev
        supervised_std = _get_moisture_stddev_between_models(supervised_models, t3, incidence) * 100
        im = ax_sup.imshow(supervised_std, extent=data_extent, **stddev_style)
        # hybrid stddev
        hybrid_std = _get_moisture_stddev_between_models(hybrid_models, t3, incidence) * 100
        im = ax_hyb.imshow(hybrid_std, extent=data_extent, **stddev_style)
        # self-supervised autoencoder stddev
        unsupervised_std = _get_moisture_stddev_between_models(unsupervised_models, t3, incidence) * 100
        im = ax_self.imshow(unsupervised_std, extent=data_extent, **stddev_style)
        # physical model stddev
        physical_std = _get_moisture_stddev_between_models(physical_models, t3, incidence) * 100
        im = ax_phys.imshow(physical_std, extent=data_extent, **stddev_style)
        # polygons
        for region_name in region_names:
            geometry = regions.get_geometry_azrg(region_name, lut)
            for ax in [ax_pauli, ax_sup, ax_hyb, ax_self, ax_phys]:
                plot_functions.plot_polygon_into_axis(ax, geometry, flip_xy=True, edgecolor="black", linewidth=0.5)
        # no ticks
        for ax in [ax_pauli, ax_sup, ax_hyb, ax_self, ax_phys]:
            ax.set_xticks([])
            ax.set_yticks([])
        if row == 0:
            ax_pauli.set_title("Pauli RGB")
            ax_sup.set_title("Supervised")
            ax_hyb.set_title("Hybrid AE")
            ax_self.set_title("Self-sup. AE")
            ax_phys.set_title("Physical")
    for i, cax in enumerate(caxs):
        if i > 0:
            fig.colorbar(im, cax=cax, orientation="horizontal")
            cax.set_xticks([0, 4, 8])
        else:
            cax.axis("off")  # no colorbar under pauli
    caxs[1].set_ylabel(
        "standard\ndeviation in %",
        rotation=0,
        horizontalalignment="right",
        verticalalignment="top",
    )
    fig.savefig(ev25.get_paper_figures_folder() / f"fig_stddev_comparison_v{version}_{len(seeds)}runs.png")
    plt.close("all")


def _relative_reconstruction_error_batch(data, reconstruction):
    return torch.linalg.matrix_norm(data - reconstruction) / torch.linalg.matrix_norm(data)


def fig_explainability():
    look_mode = constants.look_mode
    seed = 0
    version = constants.code_version
    band = constants.band
    goal_px_size = 300
    label_pass_regions = get_selected_regions_and_passes()
    regions = ev25.EarthVision2025Regions()
    # prepare inversion
    hybrid_path = ev25.get_hybrid_model_path(look_mode, version, seed)
    hybrid_model = ev25.load_encoder(hybrid_path)
    decoder = ev25.PhysicalDecoder()
    m_s, plant_mst, phi = constants.m_s, constants.plant_mst, constants.phi
    sand, clay, frequency = constants.sand, constants.clay, constants.frequency
    # prepare figure
    fig, axs, caxs = _grid_layout_bottom_cb(num_rows=len(label_pass_regions), num_cols=4)
    cax_pauli, cax_pow, cax_sm, cax_err = caxs
    moisture_style = dict(vmin=0, vmax=50, cmap="viridis_r", origin="lower", aspect="auto")
    error_style = dict(vmin=0, vmax=100, cmap="Reds", origin="lower", aspect="auto")
    for row, (label, pass_name, region_names) in enumerate(label_pass_regions):
        campaign = campaign_data.get_campaign(pass_name)
        lut = campaign.get_pass(pass_name, band).load_gtc_sr2geo_lut()
        ax_pauli, ax_pow, ax_sm, ax_err = axs[row]
        t3, incidence, data_extent = _load_downsampled_data(pass_name, band, region_names, look_mode, goal_px_size)
        # pauli preview
        _plot_pauli_preview_into_axis(pass_name, band, region_names, ax_pauli)
        ax_pauli.set_ylabel(f"{label}")
        cax_pauli.axis("off")
        # component powers
        m_d, m_v, soil_mst, delta = hybrid_model.forward(t3_batch=t3, incidence_batch=incidence)
        xbragg_power, dihedral_power, volume_power = decoder.get_component_powers_np(
            m_s=m_s,
            m_d=m_d,
            m_v=m_v,
            soil_mst=soil_mst,
            plant_mst=plant_mst,
            delta=delta,
            phi=phi,
            incidence=incidence,
            sand=sand,
            clay=clay,
            frequency=frequency,
        )
        power_sum = xbragg_power + dihedral_power + volume_power
        r = dihedral_power / power_sum
        g = volume_power / power_sum
        b = xbragg_power / power_sum
        comp_rgb = np.clip(np.stack([r, g, b], axis=-1), a_min=0, a_max=1)
        comp_rgb[np.isnan(comp_rgb)] = 0
        ax_pow.imshow(comp_rgb, extent=data_extent, origin="lower", aspect="auto")
        cax_pow.axis("off")
        # soil moisture
        soil_mst_np = soil_mst.cpu().detach().numpy() * 100
        im_sm = ax_sm.imshow(soil_mst_np, extent=data_extent, **moisture_style)
        fig.colorbar(im_sm, cax=cax_sm, orientation="horizontal")
        # reconstruction error
        reconstruction = decoder.forward(
            m_s=m_s,
            m_d=m_d,
            m_v=m_v,
            soil_mst=soil_mst,
            plant_mst=plant_mst,
            delta=delta,
            phi=phi,
            incidence=incidence,
            sand=sand,
            clay=clay,
            frequency=frequency,
        )
        rec_err = _relative_reconstruction_error_batch(t3, reconstruction).cpu().detach().numpy() * 100
        im_err = ax_err.imshow(rec_err, extent=data_extent, **error_style)
        fig.colorbar(im_err, cax=cax_err, orientation="horizontal")
        # polygons
        for region_name in region_names:
            geometry = regions.get_geometry_azrg(region_name, lut)
            for ax in [ax_pauli, ax_pow, ax_sm, ax_err]:
                plot_functions.plot_polygon_into_axis(ax, geometry, flip_xy=True, edgecolor="black", linewidth=0.5)
        # no ticks
        for ax in [ax_pauli, ax_pow, ax_sm, ax_err]:
            ax.set_xticks([])
            ax.set_yticks([])
        if row == 0:
            ax_pauli.set_title("Pauli RGB")
            ax_pow.set_title("Comp. powers")
            ax_sm.set_title("Moisture")
            ax_err.set_title("Reconstr. error")
    cax_sm.set_xticks([5, 25, 45])
    cax_err.set_xticks([0, 40, 80])
    cax_sm.set_ylabel(
        "soil moisture in %\nreconstruction error in %",
        rotation=0,
        horizontalalignment="right",
        verticalalignment="top",
    )
    fig.savefig(ev25.get_paper_figures_folder() / f"fig_explainability_v{version}s{seed}.png")
    plt.close("all")


def look_statistics_and_resolution():
    band = constants.band
    look_mode = constants.look_mode
    for pass_name in ["14cropex0210", "14cropex1318", "22hterra0104", "22hterra0504"]:
        pixels_az, pixels_rg = campaign_data.get_window_size(pass_name, band, look_mode)
        campaign = campaign_data.get_campaign(pass_name)
        fsar_pass = campaign.get_pass(pass_name, band)
        rdp_params = fsar_pass.load_rgi_params()
        looks = fc.convert_pixels_to_looks(rdp_params, pixels_az, pixels_rg)
        meters = fc.convert_pixels_to_meters(rdp_params, pixels_az, pixels_rg)
        pixels_str = f"{pixels_az} x {pixels_rg} pixels"
        looks_str = f"{looks[0] * looks[1]:.2f} looks"
        meters_str = f"{meters[0]:.1f} x {meters[1]:.1f} meters"
        print(f"Pass {pass_name}: {pixels_str}, {looks_str}, {meters_str}")


def main_paper_figures():
    setup_matplotlib_paper_figure_styles()
    fig_labeled_dataset()
    fig_fsar_pauli_slc()
    fig_moisture_comparison()
    fig_train_val_test_scatter()
    fig_ood_scatter()
    fig_stddev_comparison()
    fig_explainability()
    look_statistics_and_resolution()


if __name__ == "__main__":
    main_paper_figures()
