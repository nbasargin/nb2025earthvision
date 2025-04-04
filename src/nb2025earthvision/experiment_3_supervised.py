"""
Supervised model

Learn a function from polarimetric coherency matrices to soil moisture.
Use CROPEX data over maize fields for training, validation, and testing.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import nb2025earthvision as ev25
from nb2025earthvision import datasets, plot_functions, validation, constants


def _train_supervised(band, look_mode, lr, iterations, seed):
    dataset_folder = ev25.get_dataset_folder()
    train_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TRAIN, band, look_mode, dataset_folder)
    val_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_VAL, band, look_mode, dataset_folder)
    test_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TEST, band, look_mode, dataset_folder)
    torch.manual_seed(seed)
    model = ev25.CoherencyEncoder()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    train_rmse_hist, val_rmse_hist, test_rmse_hist = [], [], []
    for iteration in range(iterations):
        optimizer.zero_grad()
        prediction = model.predict_soil_moisture(train_ds.t3_tensors, train_ds.inc_tensors)
        loss = criterion(prediction, train_ds.sm_tensors)
        loss.backward()
        optimizer.step()
        if iteration % 100 == 99 or iteration + 1 == iterations:
            print(f"iteration {iteration + 1:5} / {iterations}: loss {loss.item():.7f}")
        # track performance on train, validation, and test datasets
        train_rmse_hist.append(validation.get_moisture_rmse_on_dataset(model, train_ds))
        val_rmse_hist.append(validation.get_moisture_rmse_on_dataset(model, val_ds))
        test_rmse_hist.append(validation.get_moisture_rmse_on_dataset(model, test_ds))
    return model, train_rmse_hist, val_rmse_hist, test_rmse_hist


def main_supervised(seed):
    version = constants.code_version
    band = constants.band
    look_mode = constants.look_mode
    supervised_model_path = ev25.get_supervised_model_path(look_mode, version, seed)
    out_folder = ev25.get_supplementary_figures_folder()
    if supervised_model_path.exists():
        print(f"Supervised model aready exists for {look_mode} s{seed}, skip training")
        return

    version_seed = f"{look_mode}_v{version}s{seed}"
    max_iterations = 30000
    lr = 0.001
    print(f"Starting supervised training {version_seed}")

    # train for max number of iterations
    model_max, train_rmse_hist_max, val_rmse_hist_max, test_rmse_hist_max = _train_supervised(
        band=band,
        look_mode=look_mode,
        lr=lr,
        iterations=max_iterations,
        seed=seed,
    )
    best_validation_rmse = np.min(val_rmse_hist_max)
    optimal_iter = np.argmin(val_rmse_hist_max)
    print(f"Best iteration is {optimal_iter} with validation RMSE of {best_validation_rmse * 100:.2f} vol%")
    plot_functions.plot_parameter_history(
        {
            "train RMSE": np.array(train_rmse_hist_max) * 100,
            "validation RMSE": np.array(val_rmse_hist_max) * 100,
            "test RMSE": np.array(test_rmse_hist_max) * 100,
        },
        xlabel="iteration",
        ylabel="moisture RMSE in vol.%",
        ylim=(0, 10),
        title=f"Supervised, RMSE by iteration {version_seed}, optimal iteration {optimal_iter}",
        save_to=out_folder / f"{supervised_model_path.stem}__1_rmse_history_max_iter.jpg",
    )

    # train until the optimal iteration
    model_opt, train_rmse_hist_opt, val_rmse_hist_opt, test_rmse_hist_opt = _train_supervised(
        seed=seed,
        band=band,
        look_mode=look_mode,
        lr=lr,
        iterations=optimal_iter,
    )
    plot_functions.plot_parameter_history(
        {
            "train RMSE": np.array(train_rmse_hist_opt) * 100,
            "validation RMSE": np.array(val_rmse_hist_opt) * 100,
            "test RMSE": np.array(test_rmse_hist_opt) * 100,
        },
        xlabel="iteration",
        ylabel="moisture RMSE in vol.%",
        ylim=(0, 10),
        title=f"Supervised, RMSE by iteration {version_seed}, optimal iteration {optimal_iter}",
        save_to=out_folder / f"{supervised_model_path.stem}__2_rmse_history_optimal_iter.jpg",
    )

    # save model
    ev25.save_encoder(model_opt, supervised_model_path)

    # quick validation on labelled datasets
    dataset_folder = ev25.get_dataset_folder()
    train_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TRAIN, band, look_mode, dataset_folder)
    val_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_VAL, band, look_mode, dataset_folder)
    test_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TEST, band, look_mode, dataset_folder)
    validation.evaluate_model_on_supervised_datasets(
        model=model_opt,
        dataset_dict={"train": train_ds, "val": val_ds, "test": test_ds},
        title=f"Supervised model {version_seed}",
        save_to=out_folder / f"{supervised_model_path.stem}__9_predictions.jpg",
    )


if __name__ == "__main__":
    main_supervised(seed=0)
