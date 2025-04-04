import pathlib


def get_root_folder():
    folder = pathlib.Path("nb2025earthvision_results")
    folder.mkdir(exist_ok=True, parents=True)
    return folder


def get_dataset_folder():
    folder = get_root_folder() / "datasets"
    folder.mkdir(exist_ok=True, parents=True)
    return folder


def get_model_folder():
    folder = get_root_folder() / "models"
    folder.mkdir(exist_ok=True, parents=True)
    return folder


def get_supplementary_figures_folder():
    folder = get_root_folder() / "supplementary_figures"
    folder.mkdir(exist_ok=True, parents=True)
    return folder


def get_paper_figures_folder():
    folder = get_root_folder() / "paper_figures"
    folder.mkdir(exist_ok=True, parents=True)
    return folder


def get_supervised_model_path(look_mode, version, seed):
    return get_model_folder() / f"earthvision25_model_supervised_{look_mode}_v{version}s{seed}.pth"


def get_selfsupervised_model_path(look_mode, version, seed):
    return get_model_folder() / f"earthvision25_model_selfsupervised_{look_mode}_v{version}s{seed}.pth"


def get_hybrid_model_path(look_mode, version, seed):
    return get_model_folder() / f"earthvision25_model_hybrid_{look_mode}_v{version}s{seed}.pth"
