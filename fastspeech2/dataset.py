"""
Dataset and DataLoader utilities for FastSpeech 2.
Handles text tokenization, mel, pitch, energy, and duration alignment.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from .utils import pad_1D, pad_2D
from .text import text_to_sequence


class FastSpeech2Dataset(Dataset):
    def __init__(self, data_list, language="English", stats=None):
        """
        data_list: list of dicts with keys:
            id, text, mel, pitch, energy, duration
        """
        self.data_list = data_list
        self.language = language
        self.stats = stats or {
            "pitch": [60.0, 450.0, 200.0, 50.0],
            "energy": [0.0, 100.0, 30.0, 15.0],
        }

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        uid = item.get("id", f"utt_{idx}")
        text = item["text"]
        seq, _ = text_to_sequence(text, self.language)
        seq = np.array(seq, dtype=np.int64)

        mel = item.get("mel", None)
        pitch = item.get("pitch", None)
        energy = item.get("energy", None)
        duration = item.get("duration", None)

        # If synthetic / training without real preprocessed arrays, generate aligned targets
        T_text = len(seq)
        if duration is None:
            # Approx 3 frames per phoneme on average (256 hop @ 22050 Hz ~ 11.6ms/frame -> ~35ms/phoneme)
            duration = np.random.randint(2, 5, size=T_text, dtype=np.int64)
            # Make pauses slightly longer
            duration[0] = 5
            duration[-1] = 6

        T_mel = int(np.sum(duration))
        if mel is None:
            # Synthetic 80-channel mel representation
            mel = np.random.randn(T_mel, 80).astype(np.float32) * 0.5 - 4.0
        if pitch is None:
            # Realistic pitch contour around mean pitch
            t = np.linspace(0, np.pi, T_text)
            pitch = (180.0 + 35.0 * np.sin(t) + np.random.randn(T_text) * 5.0).astype(np.float32)
        if energy is None:
            energy = (25.0 + 10.0 * np.sin(t) + np.random.randn(T_text) * 2.0).astype(np.float32)

        return {
            "id": uid,
            "text": text,
            "speaker": 0,
            "seq": seq,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

    @staticmethod
    def collate_fn(batch):
        ids = [b["id"] for b in batch]
        raw_texts = [b["text"] for b in batch]
        speakers = torch.tensor([b["speaker"] for b in batch], dtype=torch.long)

        # Phoneme sequence padding
        seqs = [b["seq"] for b in batch]
        src_lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        max_src_len = max(len(s) for s in seqs)
        padded_seqs = torch.tensor(pad_1D(seqs, PAD=0), dtype=torch.long)

        # Duration, pitch, energy padding
        durations = [b["duration"] for b in batch]
        padded_durations = torch.tensor(pad_1D(durations, PAD=0), dtype=torch.long)

        pitches = [b["pitch"] for b in batch]
        padded_pitches = torch.tensor(pad_1D(pitches, PAD=0), dtype=torch.float32)

        energies = [b["energy"] for b in batch]
        padded_energies = torch.tensor(pad_1D(energies, PAD=0), dtype=torch.float32)

        # Mel spectrogram padding
        mels = [b["mel"] for b in batch]
        mel_lens = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
        max_mel_len = max(m.shape[0] for m in mels)
        padded_mels = torch.tensor(pad_2D(mels, max_mel_len, PAD=0), dtype=torch.float32)

        return (
            ids,
            raw_texts,
            speakers,
            padded_seqs,
            src_lens,
            max_src_len,
            padded_mels,
            mel_lens,
            max_mel_len,
            padded_pitches,
            padded_energies,
            padded_durations,
        )

