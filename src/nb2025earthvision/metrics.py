import torch
from torchmetrics.functional import mean_squared_error


def mean_squared_error_matrix(matrix_batch_1, matrix_batch_2):
    batch_squared_errors = torch.sum(torch.abs(matrix_batch_1 - matrix_batch_2) ** 2, dim=(-2, -1))
    mean_error = torch.mean(batch_squared_errors)
    return mean_error


def get_rmse(prediction, target):
    return torch.sqrt(mean_squared_error(prediction, target)).item()


def get_bias(prediction, target):
    return torch.mean(prediction - target)
