"""
Physical model inversion

Directly invert physical parameters using optimization.
"""

import torch

import nb2025earthvision as ev25
from nb2025earthvision import validation, constants, datasets


def main_physical(seed):
    torch.manual_seed(seed)
    version = constants.code_version
    band = constants.band
    look_mode = constants.look_mode
    version_seed = f"{look_mode}_v{version}s{seed}"
    print(f"Start physical model inversion {version_seed}")
    out_folder = ev25.get_supplementary_figures_folder()
    model = ev25.PhysicalInversionModel(
        m_s=constants.m_s,
        plant_mst=constants.plant_mst,
        phi=constants.phi,
        sand=constants.sand,
        clay=constants.clay,
        frequency=constants.frequency,
        seed=seed,
    )
    # evaluate model on supervised datasets
    dataset_folder = ev25.get_dataset_folder()
    train_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TRAIN, band, look_mode, dataset_folder)
    val_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_VAL, band, look_mode, dataset_folder)
    test_ds = datasets.get_labeled_dataset_by_id(datasets.CROPEX_MA_TEST, band, look_mode, dataset_folder)
    validation.evaluate_model_on_supervised_datasets(
        model=model,
        dataset_dict={"train": train_ds, "val": val_ds, "test": test_ds},
        title=f"Physical model {version_seed}",
        save_to=out_folder / f"earthvision25_model_physical_{version_seed}__9_predictions.jpg",
    )


if __name__ == "__main__":
    main_physical(seed=0)
