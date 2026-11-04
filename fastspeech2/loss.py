"""
FastSpeech 2 Multi-Task Loss Function.
Combines Mel Spectrogram L1 Loss (Pre- and Post-Net),
Pitch MSE, Energy MSE, and Duration MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        (
            mel_before,
            mel_after,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            _,
        ) = predictions

        (
            _,
            _,
            p_targets,
            e_targets,
            d_targets,
            mel_targets,
        ) = targets

        # Duration loss: targets are log(duration + 1)
        log_d_targets = torch.log(d_targets.float() + 1.0)
        log_d_targets = log_d_targets.masked_select(~src_masks)
        log_d_predictions = log_d_predictions.masked_select(~src_masks)
        duration_loss = self.mse_loss(log_d_predictions, log_d_targets)

        # Pitch loss
        p_targets = p_targets.masked_select(~src_masks)
        p_predictions = p_predictions.masked_select(~src_masks)
        pitch_loss = self.mse_loss(p_predictions, p_targets)

        # Energy loss
        e_targets = e_targets.masked_select(~src_masks)
        e_predictions = e_predictions.masked_select(~src_masks)
        energy_loss = self.mse_loss(e_predictions, e_targets)

        # Mel loss
        # Slice to common length if different
        min_mel_len = min(mel_after.shape[1], mel_targets.shape[1])
        mel_before = mel_before[:, :min_mel_len, :]
        mel_after = mel_after[:, :min_mel_len, :]
        mel_targets = mel_targets[:, :min_mel_len, :]
        valid_mel_masks = mel_masks[:, :min_mel_len]

        mel_before = mel_before.masked_select((~valid_mel_masks).unsqueeze(-1))
        mel_after = mel_after.masked_select((~valid_mel_masks).unsqueeze(-1))
        mel_targets = mel_targets.masked_select((~valid_mel_masks).unsqueeze(-1))

        mel_loss_before = self.mae_loss(mel_before, mel_targets)
        mel_loss_after = self.mae_loss(mel_after, mel_targets)

        total_loss = (
            mel_loss_before + mel_loss_after + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss_before,
            mel_after,
            pitch_loss,
            energy_loss,
            duration_loss,
        )

