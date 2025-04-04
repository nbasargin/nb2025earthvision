"""
Self-supervised autoencoder

Encoder is a ML model that predicts physical parameters from PolSAR data.
Decoder is a physical model that reconstructs the data from the physical parameters.
"""

from datetime import datetime
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import nb2025earthvision as ev25
from nb2025earthvision import datasets, validation, constants


def _plot_iteration_and_epoch_loss(epoch_loss, iter_loss, title, save_to, ylim=(None, None)):
    fig, axs = plt.subplots(nrows=2, figsize=(8, 7))
    fig.subplots_adjust(left=0.12, bottom=0.1, right=0.92, top=0.88)
    fig.suptitle(title)
    # epochs
    axs[0].plot(epoch_loss)
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("loss")
    axs[0].set_ylim(*ylim)
    axs[0].grid()
    # iterations
    axs[1].plot(iter_loss)
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel("loss")
    axs[1].set_ylim(*ylim)
    axs[1].grid()
    fig.savefig(save_to, dpi=300)
    plt.close("all")


def _train_selfsupervised(
    encoder: ev25.CoherencyEncoder,
    decoder: ev25.PhysicalDecoder,
    dataset: datasets.UnlabeledDataset,
    epochs,
    batch_size,
    lr,
    scheduler_step_size,
    scheduler_gamma,
):
    start = datetime.now()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer_encoder = Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=scheduler_step_size, gamma=scheduler_gamma)
    epoch_loss_hist = []
    iter_loss_hist = []
    for epoch in range(epochs):
        for batch_t3s, batch_incs in loader:
            optimizer_encoder.zero_grad()
            batch_m_d, batch_m_v, batch_soil_mst, batch_delta = encoder.forward(
                t3_batch=batch_t3s, incidence_batch=batch_incs
            )
            reconstruction = decoder.forward(
                m_s=constants.m_s,
                m_d=batch_m_d,
                m_v=batch_m_v,
                soil_mst=batch_soil_mst,
                plant_mst=constants.plant_mst,
                delta=batch_delta,
                phi=constants.phi,
                incidence=batch_incs,
                sand=constants.sand,
                clay=constants.clay,
                frequency=constants.frequency,
            )
            loss = ev25.mean_squared_error_matrix(reconstruction, batch_t3s)
            loss.backward()
            optimizer_encoder.step()
            iter_loss_hist.append(loss.item())
        scheduler.step()
        # track performance after epoch
        full_m_d, full_m_v, full_soil_mst, full_delta = encoder.forward(
            t3_batch=dataset.t3_tensors, incidence_batch=dataset.inc_tensors
        )
        full_reconstruction = decoder.forward(
            m_s=constants.m_s,
            m_d=full_m_d,
            m_v=full_m_v,
            soil_mst=full_soil_mst,
            plant_mst=constants.plant_mst,
            delta=full_delta,
            phi=constants.phi,
            incidence=dataset.inc_tensors,
            sand=constants.sand,
            clay=constants.clay,
            frequency=constants.frequency,
        )
        loss = ev25.mean_squared_error_matrix(full_reconstruction, dataset.t3_tensors)
        epoch_loss_hist.append(loss.item())
        print(f"Epoch {epoch+1:>3} total loss = {loss.item():.7f}")

    sec_passed = (datetime.now() - start).total_seconds()
    print(f"Self-supervised training complete in {sec_passed:.2f} seconds")
    return iter_loss_hist, epoch_loss_hist


def main_selfsupervised(seed):
    version = constants.code_version
    band = constants.band
    look_mode = constants.look_mode
    selfsupervised_model_path = ev25.get_selfsupervised_model_path(look_mode, version, seed)
    out_folder = ev25.get_supplementary_figures_folder()
    if selfsupervised_model_path.exists():
        print(f"Self-supervised model aready exists for {look_mode} s{seed}, skip training")
        return

    version_seed = f"{look_mode}_v{version}s{seed}"
    print(f"Starting self-supervised training {version_seed}")
    epochs = 60
    batch_size = 2**6
    lr = 0.002
    scheduler_step_size = 40
    scheduler_gamma = 0.1
    loss_ylim = (0.000, 0.01)

    # models and datasets
    torch.manual_seed(seed)
    dataset_folder = ev25.get_dataset_folder()
    unlabeled_ds = datasets.get_unlabeled_dataset(band, look_mode, dataset_folder)
    train_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TRAIN, band, look_mode, dataset_folder)
    val_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_VAL, band, look_mode, dataset_folder)
    test_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TEST, band, look_mode, dataset_folder)
    encoder = ev25.CoherencyEncoder()
    decoder = ev25.PhysicalDecoder()
    epoch_loss_hist = []
    iter_loss_hist = []

    print("Start self-supervised training")
    iter_loss_hist, epoch_loss_hist = _train_selfsupervised(
        encoder=encoder,
        decoder=decoder,
        dataset=unlabeled_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
    )

    # plot training progress
    _plot_iteration_and_epoch_loss(
        epoch_loss_hist,
        iter_loss_hist,
        title=f"Self-supervised autoencoder loss by epoch and iteration, {version_seed}",
        save_to=out_folder / f"{selfsupervised_model_path.stem}__1_loss_history.jpg",
        ylim=loss_ylim,
    )

    # save encoder
    ev25.save_encoder(encoder, selfsupervised_model_path)

    # quick validation on labelled datasets
    validation.evaluate_model_on_supervised_datasets(
        model=encoder,
        dataset_dict={"train": train_ds, "val": val_ds, "test": test_ds},
        title=f"Self-supervised autoencoder {version_seed}",
        save_to=out_folder / f"{selfsupervised_model_path.stem}__9_predictions.jpg",
    )


if __name__ == "__main__":
    main_selfsupervised(seed=0)
