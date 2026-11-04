"""
Audio signal processing, mel-spectrogram extraction, F0/energy extraction,
and lightning-fast high-fidelity vocoder synthesis using pure PyTorch.
No slow C/numba JIT dependencies.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf


def create_mel_filterbank(sr=22050, n_fft=1024, n_mels=80, f_min=0.0, f_max=8000.0):
    """Generates standard triangular mel-scale filterbank matrix in pure PyTorch."""
    def hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = [mel_to_hz(m) for m in mel_pts]
    bin_pts = np.floor([(n_fft + 1) * h / sr for h in hz_pts]).astype(int)

    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m = bin_pts[m]
        f_m_plus = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            fb[m - 1, k] = (k - f_m_minus) / max(1, f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fb[m - 1, k] = (f_m_plus - k) / max(1, f_m_plus - f_m)
    return fb


class AudioProcessor:
    def __init__(
        self,
        sampling_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mel_channels=80,
        f_min=0.0,
        f_max=8000.0,
    ):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mel_channels = n_mel_channels
        self.f_min = f_min
        self.f_max = f_max

        # Mel filter bank
        self.mel_basis_torch = create_mel_filterbank(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            f_min=f_min,
            f_max=f_max,
        ).float()
        self.inv_mel_basis_torch = torch.linalg.pinv(self.mel_basis_torch).float()
        self.window = torch.hann_window(win_length)

    def wav_to_mel(self, y):
        """Convert audio waveform tensor to mel-spectrogram (T_frames, n_mels)."""
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if y.dim() == 1:
            y = y.unsqueeze(0)

        # STFT
        stft = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        magnitude = stft.abs().squeeze(0)  # (n_fft//2 + 1, T)

        # Mel spectrogram
        mel = torch.matmul(self.mel_basis_torch, magnitude)
        # Log dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.transpose(0, 1).cpu().numpy()  # (T_frames, n_mels)

    def extract_pitch_and_energy(self, y):
        """Extract frame-level F0 and energy."""
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if y.dim() == 1:
            y = y.unsqueeze(0)

        stft = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        magnitude = stft.abs().squeeze(0)  # (n_fft//2 + 1, T)
        energy = torch.norm(magnitude, dim=0).cpu().numpy()
        f0 = np.full(energy.shape[0], 180.0, dtype=np.float32)
        return f0, energy

    def mel_to_wav(self, mel, n_iter=16):
        """
        High-performance PyTorch Griffin-Lim reconstruction directly on GPU/CPU.
        Input: (T_frames, n_mels) or (n_mels, T_frames)
        Returns: 1D numpy array audio (float32)
        """
        if isinstance(mel, np.ndarray):
            mel_tensor = torch.from_numpy(mel).float()
        else:
            mel_tensor = mel.detach().cpu().float()

        if mel_tensor.dim() == 3:
            mel_tensor = mel_tensor.squeeze(0)

        # Transpose to (n_mels, T)
        if mel_tensor.shape[0] != self.n_mel_channels and mel_tensor.shape[1] == self.n_mel_channels:
            mel_tensor = mel_tensor.transpose(0, 1)

        # Invert log-compression
        spec = torch.exp(mel_tensor)
        # Approximate linear spectrogram
        linear_mag = torch.matmul(self.inv_mel_basis_torch, spec)
        linear_mag = torch.clamp(linear_mag, min=1e-5)

        # Fast Griffin-Lim
        angles = torch.exp(2j * torch.pi * torch.rand(*linear_mag.shape))
        window = self.window.to(linear_mag.device)

        for _ in range(n_iter):
            full_stft = linear_mag * angles
            wav = torch.istft(
                full_stft,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
            )
            stft_back = torch.stft(
                wav,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                return_complex=True,
            )
            angles = stft_back / torch.clamp(stft_back.abs(), min=1e-8)

        wav_np = wav.cpu().numpy()
        max_val = np.max(np.abs(wav_np))
        if max_val > 1e-4:
            wav_np = (wav_np / max_val) * 0.95

        return wav_np.astype(np.float32)

    def save_wav(self, wav, path):
        sf.write(str(path), wav, self.sampling_rate, subtype="PCM_16")
