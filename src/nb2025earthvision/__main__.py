from nb2025earthvision import datasets, paper_figures, experiment_1_calibration
from nb2025earthvision import experiment_2_physical, experiment_3_supervised
from nb2025earthvision import experiment_4_selfsupervised, experiment_5_hybrid


def main():
    # generate datasets, requires access to F-SAR campaign data
    datasets.main_datasets()

    # run experiments and create models, requires saved datasets
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for seed in seeds:
        print(f"Running experiments, seed {seed}")
        experiment_1_calibration.main_calibration(seed=seed)
        experiment_2_physical.main_physical(seed=seed)
        experiment_3_supervised.main_supervised(seed=seed)
        experiment_4_selfsupervised.main_selfsupervised(seed=seed)
        experiment_5_hybrid.main_hybrid(seed=seed)

    # generate paper figures, requires saved datasets and models
    paper_figures.main_paper_figures()


if __name__ == "__main__":
    main()
