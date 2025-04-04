import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pathlib
import fsarcamp as fc

import nb2025earthvision as ev25
from nb2025earthvision import constants, campaign_data

# Supervised dataset IDs
CROPEX_MA_TRAIN = "CROPEX_MA_TRAIN"
CROPEX_MA_VAL = "CROPEX_MA_VAL"
CROPEX_MA_TEST = "CROPEX_MA_TEST"
CROPEX_WH = "CROPEX_WH"
CROPEX_CU = "CROPEX_CU"
HTERRA_MA_CAIONE = "HTERRA_MA_CAIONE"
HTERRA_MA_CREA = "HTERRA_MA_CREA"
HTERRA_BS_QU = "HTERRA_BS_QU"
HTERRA_AA = "HTERRA_AA"
HTERRA_SF = "HTERRA_SF"
HTERRA_WH_CREA = "HTERRA_WH_CREA"
HTERRA_WH_CAIONE = "HTERRA_WH_CAIONE"


class LabeledDataset(Dataset):
    def __init__(self, t3_tensors: torch.Tensor, inc_tensors: torch.Tensor, sm_tensors: torch.Tensor, identifier=None):
        self.t3_tensors = t3_tensors
        self.inc_tensors = inc_tensors
        self.sm_tensors = sm_tensors
        self.identifier = identifier

    def __len__(self):
        return self.t3_tensors.shape[0]

    def __getitem__(self, idx):
        return self.t3_tensors[idx], self.inc_tensors[idx], self.sm_tensors[idx]


def _create_labeled_data(load_config: list[tuple[str, str, str]], look_mode):
    """
    Load polarimetric coherency matrices, incidence angle, and soil moisture at ground measurement positions.
    `load_config` is a list of [pass_name, band, region_name] tuples
    """
    t3_list: list[np.ndarray] = []
    inc_list: list[np.ndarray] = []
    sm_list: list[np.ndarray] = []
    for pass_name, band, region_name in load_config:
        print(f"Supervised dataset: loading {pass_name}, {band}, {region_name}")
        region_df = campaign_data.get_region_sm_points(pass_name, band, region_names=[region_name])
        if region_df.shape[0] == 0:
            raise ValueError(f"No moisture points available for {pass_name} {band} {region_name}!")
        window_size = campaign_data.get_window_size(pass_name, band, look_mode)
        az_min, az_max, rg_min, rg_max = campaign_data.get_region_slc_extent(
            pass_name, band, region_names=[region_name]
        )
        region_t3 = campaign_data.get_region_t3(pass_name, band, az_min, az_max, rg_min, rg_max, window_size)
        region_inc = campaign_data.get_region_incidence(pass_name, band, az_min, az_max, rg_min, rg_max)
        # lookup T3 and incidence values at the point positions
        az_idx = np.rint(region_df["azimuth"].to_numpy() - az_min)
        rg_idx = np.rint(region_df["range"].to_numpy() - rg_min)
        max_az, max_rg = region_inc.shape
        invalid = (
            np.isnan(az_idx) | np.isnan(rg_idx) | (az_idx < 0) | (rg_idx < 0) | (az_idx >= max_az) | (rg_idx >= max_rg)
        )
        if np.any(invalid):
            raise ValueError(f"Invalid az/rg indices detected for {pass_name} {band} {region_name}!")
        point_t3 = np.copy(fc.nearest_neighbor_lookup(region_t3, az_idx, rg_idx, inv_value=np.nan))
        point_inc = np.copy(fc.nearest_neighbor_lookup(region_inc, az_idx, rg_idx, inv_value=np.nan))
        point_sm = region_df["soil_moisture"].to_numpy().astype(np.float32)
        # check shapes
        n_points = point_sm.shape[0]
        assert point_t3.shape == (n_points, 3, 3)
        assert point_inc.shape == (n_points,)
        assert point_sm.shape == (n_points,)
        # store loaded point data in lists
        t3_list.append(point_t3)
        inc_list.append(point_inc)
        sm_list.append(point_sm)
    t3_tensors = torch.tensor(np.concatenate(t3_list, axis=0))
    inc_tensors = torch.tensor(np.concatenate(inc_list, axis=0))
    sm_tensors = torch.tensor(np.concatenate(sm_list, axis=0))
    return t3_tensors, inc_tensors, sm_tensors


def get_labeled_dataset(pass_names, band, region_names, look_mode, dataset_folder):
    dataset_version = 3
    pass_names_short = [name.replace("14cropex", "c").replace("22hterra", "h") for name in pass_names]
    pass_str = "+".join(pass_names_short)
    region_str = "+".join(region_names)
    path = (
        pathlib.Path(dataset_folder)
        / f"earthvision25_dataset_labeled_v{dataset_version}_{pass_str}_{band}_{region_str}_{look_mode}.pth"
    )
    if not path.exists():
        print(f"Caching labeled dataset to {path}")
        load_config = []
        for pass_name in pass_names:
            for region in region_names:
                load_config.append(((pass_name, band, region)))
        t3, incidence, soil_mst = _create_labeled_data(load_config, look_mode)
        # save
        torch.save({"t3": t3, "incidence": incidence, "soil_mst": soil_mst}, path)
    data_loaded = torch.load(path, weights_only=True)
    print(f"Loaded labeled dataset from {path}, {len(data_loaded['t3'])} points")
    return LabeledDataset(data_loaded["t3"], data_loaded["incidence"], data_loaded["soil_mst"])


def get_labeled_dataset_by_id(identifer: str, band: str, look_mode: str, dataset_folder):
    cropex_passes = [
        "14cropex0210",
        "14cropex0305",
        "14cropex0620",
        "14cropex0718",
        "14cropex0914",
        "14cropex1114",
        "14cropex1318",
    ]
    cropex_passes_no_flight11 = [p for p in cropex_passes if p != "14cropex1114"]
    hterra_passes_april = ["22hterra0104", "22hterra0204", "22hterra0304", "22hterra0404"]
    hterra_passes_june = ["22hterra0504", "22hterra0604", "22hterra0704", "22hterra0804"]
    pass_names, region_names = {
        CROPEX_MA_TRAIN: (cropex_passes, [ev25.CORN_C2_TRAIN]),
        CROPEX_MA_VAL: (cropex_passes, [ev25.CORN_C2_VAL]),
        CROPEX_MA_TEST: (cropex_passes, [ev25.CORN_C1]),
        CROPEX_WH: (cropex_passes_no_flight11, [ev25.WHEAT_W10]),
        CROPEX_CU: (cropex_passes, [ev25.CUCUMBERS_CU1]),
        HTERRA_MA_CAIONE: (hterra_passes_june, [ev25.CAIONE_MA]),
        HTERRA_MA_CREA: (hterra_passes_june, [ev25.CREA_MA]),
        HTERRA_BS_QU: (hterra_passes_april + hterra_passes_june, [ev25.CREA_BS_QU]),
        HTERRA_AA: (hterra_passes_june, [ev25.CAIONE_AA]),
        HTERRA_SF: (hterra_passes_june, [ev25.CREA_SF]),
        HTERRA_WH_CREA: (hterra_passes_april, [ev25.CREA_DW]),
        HTERRA_WH_CAIONE: (hterra_passes_april, [ev25.CAIONE_DW]),
    }[identifer]
    dataset = get_labeled_dataset(pass_names, band, region_names, look_mode, dataset_folder)
    dataset.identifier = identifer
    return dataset


class UnlabeledDataset(Dataset):
    def __init__(self, t3_tensors: torch.Tensor, inc_tensors: torch.Tensor):
        self.t3_tensors = t3_tensors
        self.inc_tensors = inc_tensors

    def __len__(self):
        return self.t3_tensors.shape[0]

    def __getitem__(self, idx):
        return self.t3_tensors[idx], self.inc_tensors[idx]


def _create_unlabeled_data(
    load_config: list[tuple[str, str, str]],
    look_mode: str,
    subsampling_factor,
):
    """
    Load polarimetric coherency matrices and the incidence angle in the specified regions (or full SLC extent).
    - `load_config` is a list of [pass_name, band, region_name] tuples
        the full SLC extent is loaded if region_name is None
    - `look_mode` defines the multilook mode e.g. "20looks", "80looks", "320looks"
    - `subsampling_factor` defines subsampling with respect to the multilook window size
    """
    t3_list: list[np.ndarray] = []
    inc_list: list[np.ndarray] = []
    for pass_name, band, region_name in load_config:
        print(f"Unlabeled dataset: loading {pass_name}, {band}, {region_name}")
        window_size = campaign_data.get_window_size(pass_name, band, look_mode)
        if region_name is not None:
            # load a region
            az_min, az_max, rg_min, rg_max = campaign_data.get_region_slc_extent(
                pass_name, band, region_names=[region_name]
            )
        else:
            # load the full slc extent
            az_min, az_max, rg_min, rg_max = None, None, None, None
        region_t3 = campaign_data.get_region_t3(pass_name, band, az_min, az_max, rg_min, rg_max, window_size)
        region_inc = campaign_data.get_region_incidence(pass_name, band, az_min, az_max, rg_min, rg_max)
        # subsampling - remove redundant pixels
        subsampling_az = int(window_size[0] * subsampling_factor)
        subsampling_rg = int(window_size[1] * subsampling_factor)
        region_t3 = region_t3[::subsampling_az, ::subsampling_rg]
        region_inc = region_inc[::subsampling_az, ::subsampling_rg]
        # ignore pixels exceeding certain power
        pixel_power = region_t3.real[..., 0, 0] + region_t3.real[..., 1, 1] + region_t3.real[..., 2, 2]
        valid_pixels = pixel_power <= constants.MAX_T3_POWER
        region_t3 = np.copy(region_t3[valid_pixels])
        region_inc = np.copy(region_inc[valid_pixels])
        # store loaded point data in lists
        t3_list.append(region_t3)
        inc_list.append(region_inc)
    t3_tensors = torch.tensor(np.concatenate(t3_list, axis=0))
    inc_tensors = torch.tensor(np.concatenate(inc_list, axis=0))
    return t3_tensors, inc_tensors


def get_unlabeled_dataset(band, look_mode, dataset_folder):
    dataset_version = 6
    path = pathlib.Path(dataset_folder) / f"earthvision25_dataset_unlabeled_v{dataset_version}_{band}_{look_mode}.pth"
    if not path.exists():
        print("Cache missing, creating unlabeled dataset")
        cropex_passes = [
            "14cropex0210",
            "14cropex0305",
            "14cropex0620",
            "14cropex0718",
            "14cropex0914",
            "14cropex1114",
            "14cropex1318",
        ]
        hterra_passes = [
            "22hterra0104",
            "22hterra0204",
            "22hterra0304",
            "22hterra0404",
            "22hterra0504",
            "22hterra0604",
            "22hterra0704",
            "22hterra0804",
        ]
        all_passes = [*cropex_passes, *hterra_passes]
        load_config = [(pass_name, band, None) for pass_name in all_passes]
        t3, incidence = _create_unlabeled_data(
            load_config=load_config,
            look_mode=look_mode,
            subsampling_factor=2,
        )
        torch.save({"t3": t3, "incidence": incidence}, path)
    # load from cache
    data_loaded = torch.load(path, weights_only=True)
    print(f"Loaded unlabeled dataset from {path}, {len(data_loaded['t3'])} samples")
    dataset_loaded = UnlabeledDataset(t3_tensors=data_loaded["t3"], inc_tensors=data_loaded["incidence"])
    return dataset_loaded


class InfiniteDataLoader(DataLoader):
    """Adapted from https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()  # initialize an iterator over the dataset

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()  # dataset exhausted, use a new fresh iterator
            batch = next(self.dataset_iterator)
        return batch


def main_datasets():
    dataset_folder = ev25.get_dataset_folder()
    band = constants.band
    look_mode = constants.look_mode
    # in-distribution datasets
    get_labeled_dataset_by_id(CROPEX_MA_TRAIN, band, look_mode, dataset_folder)
    get_labeled_dataset_by_id(CROPEX_MA_VAL, band, look_mode, dataset_folder)
    get_labeled_dataset_by_id(CROPEX_MA_TEST, band, look_mode, dataset_folder)
    # out-of-distribution datasets
    get_labeled_dataset_by_id(HTERRA_WH_CREA, band, look_mode, dataset_folder)
    get_labeled_dataset_by_id(HTERRA_MA_CAIONE, band, look_mode, dataset_folder)
    get_labeled_dataset_by_id(CROPEX_CU, band, look_mode, dataset_folder)
    # unsupervised dataset
    get_unlabeled_dataset(band, look_mode, dataset_folder)


if __name__ == "__main__":
    main_datasets()
