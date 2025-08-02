import numpy as np
import torch

def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Ensure we have enough data to sample from
    max_start_index = len(x) - context_length
    if max_start_index <= 0:
        raise ValueError("Input array x is too short for the given context_length.")

    # Randomly sample batch_size starting positions
    starts = np.random.randint(0, max_start_index, size=batch_size)

    # Prepare input and target sequences
    inputs = np.stack([x[start : start + context_length] for start in starts])
    targets = np.stack([x[start + 1 : start + context_length + 1] for start in starts])

    # Convert to PyTorch tensors and move to the specified device
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs_tensor, targets_tensor