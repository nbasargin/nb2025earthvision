"""
Hybrid autoencoder

Hybrid fine-tuning of a self-supervised autoencoder.
"""

from datetime import datetime
import numpy as np
import torch.nn as nn
from torch.optim import Adam

import nb2025earthvision as ev25
from nb2025earthvision import datasets, plot_functions, validation, constants


def _get_selfsupervised_loss(encoder: ev25.CoherencyEncoder, dataset: datasets.UnlabeledDataset):
    # evaluate the model
    decoder = ev25.PhysicalDecoder()
    t3, incidence = dataset.t3_tensors, dataset.inc_tensors
    m_d, m_v, soil_mst, delta = encoder.forward(t3_batch=t3, incidence_batch=incidence)
    reconstruction = decoder.forward(
        m_s=constants.m_s,
        m_d=m_d,
        m_v=m_v,
        soil_mst=soil_mst,
        plant_mst=constants.plant_mst,
        delta=delta,
        phi=constants.phi,
        incidence=incidence,
        sand=constants.sand,
        clay=constants.clay,
        frequency=constants.frequency,
    )
    loss = ev25.mean_squared_error_matrix(reconstruction, t3)
    return loss.item()


def _train_hybrid(band, look_mode, lr, iterations, version, seed):
    # datasets and models
    dataset_folder = ev25.get_dataset_folder()
    unlabeled_ds = datasets.get_unlabeled_dataset(band, look_mode, dataset_folder)
    train_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TRAIN, band, look_mode, dataset_folder)
    val_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_VAL, band, look_mode, dataset_folder)
    test_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TEST, band, look_mode, dataset_folder)
    encoder_path = ev25.get_selfsupervised_model_path(look_mode, version, seed)
    encoder: ev25.CoherencyEncoder = ev25.load_encoder(encoder_path)
    decoder = ev25.PhysicalDecoder()
    selfsupervised_loss_start = _get_selfsupervised_loss(encoder, unlabeled_ds)
    batch_size = train_ds.inc_tensors.shape[0]
    infinite_unlabeled_loader = datasets.InfiniteDataLoader(
        unlabeled_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    optimizer = Adam(encoder.parameters(), lr=lr)
    moisture_criterion = nn.MSELoss()
    rec_u_loss_hist, rec_l_loss_hist, moisture_s_loss_hist = [], [], []
    train_rmse_hist, val_rmse_hist, test_rmse_hist = [], [], []

    start = datetime.now()
    for iteration in range(iterations):
        t3_u, inc_u = next(infinite_unlabeled_loader)
        optimizer.zero_grad()
        # reconstruct unlabeled
        m_d_u, m_v_u, soil_mst_u, delta_u = encoder.forward(t3_batch=t3_u, incidence_batch=inc_u)
        reconstruction_u = decoder.forward(
            m_s=constants.m_s,
            m_d=m_d_u,
            m_v=m_v_u,
            soil_mst=soil_mst_u,
            plant_mst=constants.plant_mst,
            delta=delta_u,
            phi=constants.phi,
            incidence=inc_u,
            sand=constants.sand,
            clay=constants.clay,
            frequency=constants.frequency,
        )
        # reconstruct labeled
        t3_l = train_ds.t3_tensors
        inc_l = train_ds.inc_tensors
        labels_l = train_ds.sm_tensors
        m_d_l, m_v_l, soil_mst_l, delta_l = encoder.forward(t3_batch=t3_l, incidence_batch=inc_l)
        reconstruction_s = decoder.forward(
            m_s=constants.m_s,
            m_d=m_d_l,
            m_v=m_v_l,
            soil_mst=soil_mst_l,
            plant_mst=constants.plant_mst,
            delta=delta_l,
            phi=constants.phi,
            incidence=inc_l,
            sand=constants.sand,
            clay=constants.clay,
            frequency=constants.frequency,
        )
        # compute hybrid loss
        reconstruction_u_loss = ev25.mean_squared_error_matrix(reconstruction_u, t3_u)
        reconstruction_l_loss = ev25.mean_squared_error_matrix(reconstruction_s, t3_l)
        moisture_l_loss = moisture_criterion(soil_mst_l, labels_l)
        moisture_loss_weight = 0.1
        loss = reconstruction_u_loss + reconstruction_l_loss + moisture_loss_weight * moisture_l_loss
        loss.backward()
        optimizer.step()
        if iteration % 100 == 99 or iteration + 1 == iterations:
            print(f"iteration {iteration + 1:5} / {iterations}: loss {loss.item():.7f}")
        # track performance after each iteration
        rec_u_loss_hist.append(reconstruction_u_loss.item())
        rec_l_loss_hist.append(reconstruction_l_loss.item())
        moisture_s_loss_hist.append(moisture_l_loss.item())
        train_rmse_hist.append(validation.get_moisture_rmse_on_dataset(encoder, train_ds))
        val_rmse_hist.append(validation.get_moisture_rmse_on_dataset(encoder, val_ds))
        test_rmse_hist.append(validation.get_moisture_rmse_on_dataset(encoder, test_ds))
    sec_passed = (datetime.now() - start).total_seconds()
    print(f"Hybrid training complete in {sec_passed:.2f} seconds")

    selfsupervised_loss_end = _get_selfsupervised_loss(encoder, unlabeled_ds)
    print(f"self-supervised loss at the start {selfsupervised_loss_start:.7f}")
    print(f"self-supervised loss in the end {selfsupervised_loss_end:.7f}")
    return (
        encoder,
        train_rmse_hist,
        val_rmse_hist,
        test_rmse_hist,
        rec_u_loss_hist,
        rec_l_loss_hist,
        moisture_s_loss_hist,
    )


def main_hybrid(seed):
    version = constants.code_version
    band = constants.band
    look_mode = constants.look_mode
    hybrid_model_path = ev25.get_hybrid_model_path(look_mode, version, seed)
    out_folder = ev25.get_supplementary_figures_folder()
    if hybrid_model_path.exists():
        print(f"Hybrid model aready exists for {look_mode} s{seed}, skip training")
        return

    version_seed = f"{look_mode}_v{version}s{seed}"
    print(f"Start hybrid training {version_seed}")
    band = constants.band
    look_mode = constants.look_mode
    max_iterations = 60000
    lr = 0.0001

    # models and datasets
    (
        encoder_max,
        train_rmse_hist,
        val_rmse_hist,
        test_rmse_hist,
        rec_u_loss_hist,
        rec_l_loss_hist,
        moisture_l_loss_hist,
    ) = _train_hybrid(band, look_mode, lr, max_iterations, version, seed)

    # find iteration with best validation rmse
    best_validation_rmse = np.min(val_rmse_hist)
    optimal_iter = np.argmin(val_rmse_hist)
    print(f"Best iteration is {optimal_iter} with validation RMSE of {best_validation_rmse * 100:.2f} vol%")

    # plot validation for long training
    plot_functions.plot_parameter_history(
        {
            "train RMSE": np.array(train_rmse_hist) * 100,
            "validation RMSE": np.array(val_rmse_hist) * 100,
            "test RMSE": np.array(test_rmse_hist) * 100,
        },
        xlabel="iteration",
        ylabel="moisture RMSE in vol.%",
        ylim=(0, 10),
        title=f"Hybrid, RMSE by iteration {version_seed}, optimal iter {optimal_iter}",
        save_to=out_folder / f"{hybrid_model_path.stem}__1_rmse_history_max_{max_iterations}iters.jpg",
    )
    plot_functions.plot_parameter_history(
        {
            "reconstruction U": np.array(rec_u_loss_hist),
            "reconstruction L": np.array(rec_l_loss_hist),
            "moisture L": np.array(moisture_l_loss_hist),
        },
        xlabel="iteration",
        ylabel="loss",
        ylim=(0, 0.03),
        title=f"Hybrid, Loss components by iteration, {version_seed}",
        save_to=out_folder / f"{hybrid_model_path.stem}__3_iteration_loss_components_max_{max_iterations}iters.jpg",
        alpha=0.5,
    )

    # retrain from scratch till optimal iteration
    (
        encoder_opt,
        train_rmse_hist,
        val_rmse_hist,
        test_rmse_hist,
        rec_u_loss_hist,
        rec_l_loss_hist,
        moisture_l_loss_hist,
    ) = _train_hybrid(band, look_mode, lr, optimal_iter, version, seed)
    # plot validation for iteration training
    plot_functions.plot_parameter_history(
        {
            "train RMSE": np.array(train_rmse_hist) * 100,
            "validation RMSE": np.array(val_rmse_hist) * 100,
            "test RMSE": np.array(test_rmse_hist) * 100,
        },
        xlabel="iteration",
        ylabel="moisture RMSE in vol.%",
        ylim=(0, 10),
        title=f"Hybrid, RMSE by iteration {version_seed}, optimal iter {optimal_iter}",
        save_to=out_folder / f"{hybrid_model_path.stem}__2_rmse_history_opt_{optimal_iter}iters.jpg",
    )
    plot_functions.plot_parameter_history(
        {
            "reconstruction U": np.array(rec_u_loss_hist),
            "reconstruction L": np.array(rec_l_loss_hist),
            "moisture L": np.array(moisture_l_loss_hist),
        },
        xlabel="iteration",
        ylabel="loss",
        ylim=(0, 0.03),
        title=f"Hybrid, Loss components by iteration, {version_seed}",
        save_to=out_folder / f"{hybrid_model_path.stem}__4_iteration_loss_components_opt_{optimal_iter}iters.jpg",
        alpha=0.5,
    )

    # save fine-tuned encoder
    ev25.save_encoder(encoder_opt, hybrid_model_path)

    # quick validation on labelled datasets
    dataset_folder = ev25.get_dataset_folder()
    train_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TRAIN, band, look_mode, dataset_folder)
    val_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_VAL, band, look_mode, dataset_folder)
    test_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TEST, band, look_mode, dataset_folder)
    validation.evaluate_model_on_supervised_datasets(
        model=encoder_opt,
        dataset_dict={"train": train_ds, "val": val_ds, "test": test_ds},
        title=f"Hybrid model {version_seed}",
        save_to=out_folder / f"{hybrid_model_path.stem}__9_predictions.jpg",
    )


if __name__ == "__main__":
    main_hybrid(seed=0)
