import torch
import torch.nn.functional as F
import numpy as np


def get_mask_from_lengths(lengths, max_len=None):
    """Create boolean mask from sequence lengths. Shape: (batch_size, max_len)."""
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return ~mask  # True where padded (for PyTorch attention mask conventions)


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])
    return padded


def pad_2D(inputs, max_len=None, PAD=0):
    def pad(x, max_len):
        PAD_size = max_len - x.shape[0]
        return np.pad(x, ((0, PAD_size), (0, 0)), mode="constant", constant_values=PAD)

    if max_len is None:
        max_len = max((x.shape[0] for x in inputs))
    padded = np.stack([pad(x, max_len) for x in inputs])
    return padded

