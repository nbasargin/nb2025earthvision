"""
Physical model calibration

Find fixed values for the calibrated physical parameters using measuremetns from the train set.
"""

from datetime import datetime
import torch
import torch.nn as nn
import json

import nb2025earthvision as ev25
from nb2025earthvision import plot_functions, constants, datasets


class HyperparameterCalibration(nn.Module):
    """
    Hyperparameters to be calibrated on a dataset.
    """

    def __init__(self, dataset: datasets.LabeledDataset):
        super().__init__()
        *batch_shape, u, v = dataset.t3_tensors.shape
        assert u == 3 and v == 3
        self.batch_shape = batch_shape
        self.soil_mst = dataset.sm_tensors  # soil moisture ground measurements
        # hyperparameters to be calibrated
        self.m_s = nn.Parameter(torch.rand(()))
        self.phi = nn.Parameter(torch.rand(()) * (constants.phi_max - constants.phi_min) + constants.phi_min)
        self.plant_mst = nn.Parameter(
            torch.rand(()) * (constants.plant_mst_max - constants.plant_mst_min) + constants.plant_mst_min
        )
        # free parameters
        self.m_d = nn.Parameter(torch.rand(batch_shape))
        self.m_v = nn.Parameter(torch.rand(batch_shape))
        self.delta = nn.Parameter(
            torch.rand(batch_shape) * (constants.delta_max - constants.delta_min) + constants.delta_min
        )


def _calibrate_hyperparameters(dataset: datasets.LabeledDataset, iterations, lr, scheduler_step_size, scheduler_gamma):
    calibrated_params = HyperparameterCalibration(dataset)
    decoder = ev25.PhysicalDecoder()
    incidence = dataset.inc_tensors
    # prepare fitting
    optimizer = torch.optim.Adam(calibrated_params.parameters(), lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    loss_history = []
    start = datetime.now()
    for i in range(iterations):
        optimizer.zero_grad()
        reconstr = decoder.forward(
            m_s=calibrated_params.m_s,
            m_d=calibrated_params.m_d,
            m_v=calibrated_params.m_v,
            soil_mst=calibrated_params.soil_mst,
            plant_mst=calibrated_params.plant_mst,
            delta=calibrated_params.delta,
            phi=calibrated_params.phi,
            incidence=incidence,
            sand=constants.sand,
            clay=constants.clay,
            frequency=constants.frequency,
        )
        loss = ev25.mean_squared_error_matrix(dataset.t3_tensors, reconstr)
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            # ensure parameters stay in the valid range
            calibrated_params.plant_mst.clamp_(min=constants.plant_mst_min, max=constants.plant_mst_max)
            calibrated_params.delta.clamp_(min=constants.delta_min, max=constants.delta_max)
            calibrated_params.phi.clamp_(min=constants.phi_min, max=constants.phi_max)
        loss_history.append(loss.item())
        if i % 100 == 99:
            print(f"  iteration {i+1}  Loss = {loss.item():.7E}")
    end = datetime.now()
    print(f"  Completed {iterations} iterations in {(end - start).total_seconds():.1f} s, Loss = {loss.item():.7E}")
    return calibrated_params, loss_history


def main_calibration(seed):
    torch.manual_seed(seed)
    version = constants.code_version
    band = constants.band
    look_mode = constants.look_mode
    version_seed = f"{look_mode}_v{version}s{seed}"
    print(f"Start physical model calibration {version_seed}")
    out_folder = ev25.get_supplementary_figures_folder()

    iterations = 500
    lr = 0.3
    scheduler_step_size = 99999
    scheduler_gamma = 1
    loss_ylim = (0, 0.003)

    # evaluate model on supervised datasets
    dataset_folder = ev25.get_dataset_folder()
    train_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TRAIN, band, look_mode, dataset_folder)
    calibrated_params, loss_history = _calibrate_hyperparameters(
        dataset=train_ds,
        iterations=iterations,
        lr=lr,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
    )

    cal_m_s = calibrated_params.m_s.item()
    cal_phi_deg = torch.rad2deg(calibrated_params.phi).item()
    cal_plant_mst_percent = calibrated_params.plant_mst.item() * 100
    final_param_string = f"m_s = {cal_m_s:.3f}"
    final_param_string += f", phi = {cal_phi_deg:.2f} deg"
    final_param_string += f", plant_mst = {cal_plant_mst_percent:.2f} wgt.%"

    plot_functions.plot_parameter_history(
        {"loss": loss_history},
        ylim=loss_ylim,
        xlabel="iteration",
        ylabel="avg loss",
        title=f"Calibration, loss history, {version_seed}\n{final_param_string}",
        save_to=out_folder / f"earthvision25_calibration_{version_seed}__1_loss_history_{version_seed}.jpg",
    )

    plot_functions.plot_parameter_history(
        {"delta": torch.rad2deg(calibrated_params.delta).cpu().detach()},
        ylim=(0, 90),
        xlabel="point",
        ylabel="delta in deg",
        title=f"Calibration, delta values, {version_seed}\n{final_param_string}",
        save_to=out_folder / f"earthvision25_calibration_{version_seed}__2_delta_values.jpg",
    )

    with open(out_folder / f"earthvision25_calibration_{version_seed}__3_parameters.json", "w") as json_file:
        out_data = {
            "version": version,
            "seed": seed,
            "cal_m_s": cal_m_s,
            "cal_phi_deg": cal_phi_deg,
            "cal_plant_mst_percent": cal_plant_mst_percent,
        }
        json.dump(out_data, json_file)


if __name__ == "__main__":
    main_calibration(seed=0)
