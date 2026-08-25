"""
Utility functions for the project.
"""

from typing import Callable
import numpy.typing as npt
import gc
import numpy as np
from tqdm import tqdm
import torch


def ml_shape_repair(*arr: npt.NDArray[np.float32]):
    """
    Repairs the shape of the input data for machine learning models.
    """
    return tuple(a.squeeze() for a in arr)


def forecast_support(predict_func, x: npt.NDArray[np.float32], steps: int, **kwargs):
    """
    Support for forecast functions.
    """
    forecasted = []
    for _ in tqdm(range(steps), desc='Forecasting'):
        # Predict the output
        y = predict_func(x, **kwargs)
        # Append the output
        forecasted.append(y)
        # Update the input
        x = np.concatenate((x, y.reshape(1, -1)), axis=1)[:, 1:]
    return np.array(forecasted).squeeze()


@torch.no_grad()
def forecast_support_torch(predict_func: Callable, x: torch.Tensor, steps: int, post_func: Callable = None, **kwargs):
    """
    Support for forecast functions with torch.
    x: torch.Tensor
        Shape: (1, window_size, n_features)
    """
    forecasted = []
    for _ in tqdm(range(steps), desc='Forecasting'):
        # Predict the output
        y = predict_func(x, **kwargs)
        if post_func is not None:
            y = post_func(y)
        # Append the output
        forecasted.append(y)
        # Update the input
        x = torch.cat((x, y.unsqueeze(1)), dim=1)[:, 1:]
    return x.squeeze().detach().cpu().numpy()


def forecast_support_transformer(predict_func, x: torch.Tensor, steps: int, **kwargs):
    """
    Support for forecast functions with torch.
    """
    forecasted = []
    for _ in tqdm(range(steps), desc='Forecasting'):
        print("x shape", x.shape)
        # Predict the output
        y = predict_func(x, **kwargs)
        # Append the output
        print("y forecast shape", y.shape)
        forecasted.append(y)
        print("num of forecasted", len(forecasted))
        # Update the input
        x = torch.cat((x, y.unsqueeze(0)), dim=1)
        print("x forecast shape", x.shape)
    return torch.cat(forecasted).detach().cpu().numpy()


def forecast_support_transformer(predict_func, x: torch.Tensor, steps: int, **kwargs):
    """
    Support for forecast functions with torch.
    """
    forecasted = []
    for _ in tqdm(range(steps), desc='Forecasting'):
        # Predict the output
        y = predict_func(x, **kwargs)
        # Append the output
        forecasted.append(y)
        # Update the input
        x = torch.cat((x, y), dim=1)[:, 1:]
    return torch.cat(forecasted).squeeze().detach().cpu().numpy()


def show_table(data: list[list[str]], cols: list[str], max_length: int = 20):
    """
    Show a table in the console.
    """
    # Calculate max padding
    max_padd = [len(col) for col in cols]
    for row in data:
        for i, col in enumerate(row):
            if len(str(col)) > max_length:
                max_padd[i] = max_length
            elif len(str(col)) > max_padd[i]:
                max_padd[i] = len(str(col))
    # Trim data
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if len(str(col)) > max_length:
                data[i][j] = str(col)[:max_length - 3] + '...'
    # Show top border
    print('┌', end='')
    for i, col in enumerate(cols):
        print('─' * (max_padd[i] + 2), end='')
        if i != len(cols) - 1:
            print('┬', end='')
    print('┐')
    # Show columns
    print('│', end='')
    for i, col in enumerate(cols):
        print(f' {col}{" " * (max_padd[i] - len(str(col)))} │', end='')
    print()
    # Show divider
    print('├', end='')
    for i, col in enumerate(cols):
        print('─' * (max_padd[i] + 2), end='')
        if i != len(cols) - 1:
            print('┼', end='')
    print('┤')
    # Show content
    for row in data:
        print('│', end='')
        for i, col in enumerate(row):
            print(f' {col}{" " * (max_padd[i] - len(str(col)))} │', end='')
        print()
    # Show bottom border
    print('└', end='')
    for i, col in enumerate(cols):
        print('─' * (max_padd[i] + 2), end='')
        if i != len(cols) - 1:
            print('┴', end='')
    print('┘')


def get_available_device():
    """
    Get the available device for torch.
    """
    _device = "cpu"
    if torch.cuda.is_available():
        _device = 'cuda'
        print(f"Using device: {_device}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        _device = 'mps'
        print(f"Using device: {_device}")
    else:
        print("No available device found")
        print("Using CPU")
    return torch.device(_device)


def clear_memory():
    """
    Clear the memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        torch.mps.empty_cache()
