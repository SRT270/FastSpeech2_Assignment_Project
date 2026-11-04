"""
Variance Adaptor for FastSpeech 2:
Duration Predictor, Length Regulator, Pitch Predictor, and Energy Predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import get_mask_from_lengths


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class VariancePredictor(nn.Module):
    """
    2-layer 1D convolutional network with ReLU, LayerNorm, and Dropout,
    followed by a linear projection.
    """

    def __init__(self, hidden_dim=256, filter_size=256, kernel_size=3, dropout=0.5):
        super(VariancePredictor, self).__init__()

        self.conv1 = nn.Sequential(
            Transpose(1, 2),
            nn.Conv1d(
                hidden_dim,
                filter_size,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            Transpose(1, 2),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
        )

        self.conv2 = nn.Sequential(
            Transpose(1, 2),
            nn.Conv1d(
                filter_size,
                filter_size,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            Transpose(1, 2),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
        )

        self.linear_layer = nn.Linear(filter_size, 1)

    def forward(self, encoder_output, mask=None):
        out = self.conv1(encoder_output)
        out = self.conv2(out)
        out = self.linear_layer(out).squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class LengthRegulator(nn.Module):
    """
    Length Regulator: expands phoneme representations to match speech frame lengths.
    """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len=None):
        output = []
        mel_len = []
        for batch, d in zip(x, duration):
            expanded = self.expand(batch, d)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = self.pad(output, max_len)
        else:
            max_mel_len = max(mel_len) if mel_len else 1
            output = self.pad(output, max_mel_len)

        return output, torch.tensor(mel_len, dtype=torch.long, device=x.device)

    def expand(self, batch, predicted):
        out = []
        for i, vec in enumerate(batch):
            d = max(0, int(predicted[i].item()))
            if d > 0:
                out.append(vec.unsqueeze(0).repeat(d, 1))
        if len(out) == 0:
            return batch[:1]
        out = torch.cat(out, 0)
        return out

    def pad(self, input_ele, max_len):
        out_list = []
        for batch in input_ele:
            if batch.shape[0] < max_len:
                one_batch_padded = F.pad(
                    batch, (0, 0, 0, max_len - batch.shape[0]), "constant", 0.0
                )
            else:
                one_batch_padded = batch[:max_len]
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded

    def forward(self, x, duration, max_len=None):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VarianceAdaptor(nn.Module):
    """
    FastSpeech 2 Variance Adaptor adding Duration, Pitch, and Energy variance information.
    """

    def __init__(
        self,
        hidden_dim=256,
        filter_size=256,
        kernel_size=3,
        dropout=0.5,
        n_bins=256,
        pitch_min=60.0,
        pitch_max=450.0,
        energy_min=0.0,
        energy_max=100.0,
    ):
        super(VarianceAdaptor, self).__init__()

        self.duration_predictor = VariancePredictor(
            hidden_dim, filter_size, kernel_size, dropout
        )
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(
            hidden_dim, filter_size, kernel_size, dropout
        )
        self.energy_predictor = VariancePredictor(
            hidden_dim, filter_size, kernel_size, dropout
        )

        # Quantization bins
        self.pitch_bins = nn.Parameter(
            torch.linspace(pitch_min, pitch_max, n_bins - 1), requires_grad=False
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(energy_min, energy_max, n_bins - 1), requires_grad=False
        )

        self.pitch_embedding = nn.Embedding(n_bins, hidden_dim)
        self.energy_embedding = nn.Embedding(n_bins, hidden_dim)

    def get_pitch_embedding(self, x, target, mask, control=1.0):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            pred_controlled = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(pred_controlled, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control=1.0):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            pred_controlled = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(pred_controlled, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_mel_len=None,
        p_target=None,
        e_target=None,
        d_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        log_d_prediction = self.duration_predictor(x, src_mask)
        p_prediction, p_embedding = self.get_pitch_embedding(
            x, p_target, src_mask, p_control
        )
        e_prediction, e_embedding = self.get_energy_embedding(
            x, e_target, src_mask, e_control
        )

        # Add variance embeddings to phoneme representation
        x = x + p_embedding + e_embedding

        if d_target is not None:
            x, mel_len = self.length_regulator(x, d_target, max_mel_len)
            d_rounded = d_target
        else:
            d_rounded = torch.clamp(
                (torch.round(torch.exp(log_d_prediction) - 1) * d_control),
                min=0,
            )
            # Ensure at least 1 frame per phoneme
            d_rounded = torch.clamp(d_rounded, min=1)
            x, mel_len = self.length_regulator(x, d_rounded, max_mel_len)

        mel_mask = get_mask_from_lengths(mel_len)
        return (
            x,
            p_prediction,
            e_prediction,
            log_d_prediction,
            d_rounded,
            mel_len,
            mel_mask,
        )

