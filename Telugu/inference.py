"""
Complete Inference and Evaluation Pipeline for Telugu FastSpeech 2.
Generates 22.05 kHz audio waveforms for Telugu test sentences and produces
variance adaptor plots (Duration, Pitch/F0, Energy) for evaluation.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

# Reconfigure stdout/stderr for Unicode support on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastspeech2.model import FastSpeech2
from fastspeech2.audio import AudioProcessor
from fastspeech2.text import text_to_sequence, get_symbols


def plot_variance_components(phones, duration, pitch, energy, out_root, utt_id, text_snippet=""):
    """Plot and save individual variance components for duration, pitch, and energy."""
    out_root = Path(out_root)
    dur_dir = out_root / "duration"
    pitch_dir = out_root / "pitch"
    energy_dir = out_root / "energy"
    dur_dir.mkdir(parents=True, exist_ok=True)
    pitch_dir.mkdir(parents=True, exist_ok=True)
    energy_dir.mkdir(parents=True, exist_ok=True)

    tokens = [p.replace("@", "") for p in phones]
    x_indices = np.arange(len(tokens))

    # 1. Duration plot
    plt.figure(figsize=(11, 4.5))
    plt.bar(x_indices, duration, color="#2e7d32", edgecolor="black", alpha=0.85)
    plt.xticks(x_indices, tokens, rotation=65, ha="right", fontsize=8)
    plt.ylabel("Duration (frames)", fontsize=11)
    plt.title(f"Predicted Duration [{utt_id}]", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(dur_dir / f"{utt_id}.png", dpi=200)
    plt.close()

    # 2. Pitch (F0) plot
    plt.figure(figsize=(11, 4.5))
    plt.plot(x_indices, pitch, color="#e67e22", marker="o", lw=2, markersize=4)
    plt.xticks(x_indices, tokens, rotation=65, ha="right", fontsize=8)
    plt.ylabel("Fundamental Frequency F0 (Hz)", fontsize=11)
    plt.title(f"Predicted Pitch (F0) Contour [{utt_id}]", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(pitch_dir / f"{utt_id}.png", dpi=200)
    plt.close()

    # 3. Energy plot
    plt.figure(figsize=(11, 4.5))
    plt.plot(x_indices, energy, color="#8e44ad", marker="s", lw=2, markersize=4)
    plt.fill_between(x_indices, energy, color="#8e44ad", alpha=0.2)
    plt.xticks(x_indices, tokens, rotation=65, ha="right", fontsize=8)
    plt.ylabel("Frame Energy (L2 Norm)", fontsize=11)
    plt.title(f"Predicted Energy Distribution [{utt_id}]", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(energy_dir / f"{utt_id}.png", dpi=200)
    plt.close()


def run_inference(
    config_path="Telugu/config.yaml",
    checkpoint_path=None,
    all_tests=True,
    custom_text=None,
    pitch_control=1.0,
    energy_control=1.0,
    duration_control=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lang = cfg.get("language", "Telugu")
    ckpt_dir = Path("Telugu/checkpoints")

    if checkpoint_path is None:
        candidates = [ckpt_dir / "best_model.pth", ckpt_dir / "checkpoint_20000.pth"]
        for c in candidates:
            if c.exists():
                checkpoint_path = c
                break

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print("No checkpoint found; running training first to obtain checkpoint...")
        from Telugu.train import train
        train(config_path, num_epochs=6)
        checkpoint_path = ckpt_dir / "best_model.pth"

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stats = checkpoint.get("stats", {"pitch": [75.0, 380.0], "energy": [0.01, 92.10]})

    symbols, sym2id, _ = get_symbols(lang)
    n_vocab = len(symbols)

    model = FastSpeech2(
        n_vocab=n_vocab,
        n_mel_channels=cfg["model"]["n_mel_channels"],
        encoder_layer=cfg["model"]["encoder_layers"],
        encoder_head=cfg["model"]["encoder_heads"],
        encoder_hidden=cfg["model"]["encoder_hidden"],
        decoder_layer=cfg["model"]["decoder_layers"],
        decoder_head=cfg["model"]["decoder_heads"],
        decoder_hidden=cfg["model"]["decoder_hidden"],
        pitch_min=stats["pitch"][0],
        pitch_max=stats["pitch"][1],
        energy_min=stats["energy"][0],
        energy_max=stats["energy"][1],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    audio_proc = AudioProcessor(
        sampling_rate=cfg["model"]["sampling_rate"],
        n_fft=cfg["model"]["filter_length"],
        win_length=cfg["model"]["win_length"],
        hop_length=cfg["model"]["hop_length"],
        n_mel_channels=cfg["model"]["n_mel_channels"],
    )

    results_dir = Path("Results/Telugu")
    audio_dir = results_dir / "test_generated_audios"
    variance_dir = results_dir / "variance_adaptor_plots"
    audio_dir.mkdir(parents=True, exist_ok=True)
    variance_dir.mkdir(parents=True, exist_ok=True)

    items_to_synthesize = []
    if all_tests:
        with open("data/test_sentences.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
        items_to_synthesize = test_data["Telugu"]
    elif custom_text:
        items_to_synthesize = [{"id": "custom", "text": custom_text}]

    print(f"Synthesizing {len(items_to_synthesize)} samples...")
    for item in items_to_synthesize:
        uid = item["id"]
        raw_text = item["text"]
        print(f"Processing [{uid}]")

        seq, phones = text_to_sequence(raw_text, language=lang)
        text_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        src_len_tensor = torch.tensor([len(seq)], dtype=torch.long, device=device)

        with torch.no_grad():
            (
                mel_before,
                mel_after,
                p_pred,
                e_pred,
                log_d_pred,
                d_rounded,
                src_mask,
                mel_mask,
                mel_len,
            ) = model(
                text_tensor,
                src_len_tensor,
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
            )

        # Extract variance adaptor vectors
        dur_np = d_rounded.squeeze(0).cpu().numpy().astype(int)
        pitch_np = p_pred.squeeze(0).cpu().numpy()
        energy_np = e_pred.squeeze(0).cpu().numpy()

        # Rescale pitch and energy into physical units (Hz and RMS energy)
        pitch_hz = np.clip(pitch_np * (stats["pitch"][1] - stats["pitch"][0]) / 256.0 + stats["pitch"][0], stats["pitch"][0], stats["pitch"][1])
        energy_norm = np.clip(energy_np * (stats["energy"][1] - stats["energy"][0]) / 256.0 + stats["energy"][0], stats["energy"][0], stats["energy"][1])

        # Plot variance adaptor curves
        plot_variance_components(phones, dur_np, pitch_hz, energy_norm, variance_dir, uid, raw_text)

        # Synthesize audio waveform from postnet mel spectrogram
        mel_post_np = mel_after.squeeze(0).cpu().numpy()
        wav = audio_proc.mel_to_wav(mel_post_np)

        wav_path = audio_dir / f"{uid}.wav"
        audio_proc.save_wav(wav, wav_path)
        print(f" -> Generated: {wav_path} (length: {len(wav)/22050:.2f}s)")

    print(f"All {lang} inference tasks completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Inference for Telugu FastSpeech 2")
    parser.add_argument("--config", default="Telugu/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--all-assignment-tests", action="store_true", default=True)
    parser.add_argument("--text", default=None)
    parser.add_argument("--p-control", type=float, default=1.0)
    parser.add_argument("--e-control", type=float, default=1.0)
    parser.add_argument("--d-control", type=float, default=1.0)
    args = parser.parse_args()

    run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        all_tests=args.all_assignment_tests,
        custom_text=args.text,
        pitch_control=args.p_control,
        energy_control=args.e_control,
        duration_control=args.d_control,
    )


if __name__ == "__main__":
    main()
