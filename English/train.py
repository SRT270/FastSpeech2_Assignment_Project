"""
Training script for English FastSpeech 2 on Svarah dataset.
Executes training loop with AdamW optimizer, warmup schedule,
periodic validation, checkpoint saving, and loss curve generation.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastspeech2.model import FastSpeech2
from fastspeech2.loss import FastSpeech2Loss
from fastspeech2.dataset import FastSpeech2Dataset
from fastspeech2.text import get_symbols, text_to_sequence


def train(config_path="English/config.yaml", num_epochs=12, batch_size=8, save_plots=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lang = cfg.get("language", "English")
    print(f"=== Starting FastSpeech 2 Training for {lang} ===")

    # Load preprocessed metadata & stats
    preprocessed_path = Path("data/preprocessed/English")
    if not (preprocessed_path / "train.txt").exists():
        from English.preprocess import run_preprocessing
        run_preprocessing(config_path)

    with open(preprocessed_path / "stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    symbols, sym2id, id2sym = get_symbols(lang)
    n_vocab = len(symbols)

    # Read train items
    train_items = []
    with open(preprocessed_path / "train.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 4:
                train_items.append({"id": parts[0], "speaker": parts[1], "text": parts[3]})

    # Read validation items
    val_items = []
    with open(preprocessed_path / "val.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 4:
                val_items.append({"id": parts[0], "speaker": parts[1], "text": parts[3]})

    train_dataset = FastSpeech2Dataset(train_items, language=lang, stats=stats)
    val_dataset = FastSpeech2Dataset(val_items, language=lang, stats=stats)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=FastSpeech2Dataset.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=FastSpeech2Dataset.collate_fn,
        drop_last=False,
    )

    # Initialize model
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

    loss_fn = FastSpeech2Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-6)

    ckpt_dir = Path("English/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("Results/English")
    results_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "step": [],
        "train_loss": [],
        "mel_loss": [],
        "dur_loss": [],
        "pitch_loss": [],
        "energy_loss": [],
        "val_step": [],
        "val_loss": [],
    }

    step = 0
    best_val_loss = float("inf")

    print(f"Beginning training across {num_epochs} epochs ({len(train_loader)} steps/epoch)...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            step += 1
            (
                ids,
                raw_texts,
                speakers,
                texts,
                src_lens,
                max_src_len,
                mels,
                mel_lens,
                max_mel_len,
                pitches,
                energies,
                durations,
            ) = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]

            # Adjust learning rate with warmup schedule
            warmup_steps = 200
            lr = 256**-0.5 * min(step**-0.5, step * (warmup_steps**-1.5))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()
            predictions = model(
                texts,
                src_lens,
                max_src_len=max_src_len,
                mels=mels,
                mel_lens=mel_lens,
                max_mel_len=max_mel_len,
                p_targets=pitches,
                e_targets=energies,
                d_targets=durations,
            )

            targets = (texts, mels, pitches, energies, durations, mels)
            loss, mel_loss_pre, mel_loss_post, pitch_loss, energy_loss, dur_loss = loss_fn(
                predictions, targets
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            if step % 10 == 0 or step == 1:
                history["step"].append(step)
                history["train_loss"].append(float(loss.item()))
                history["mel_loss"].append(float(mel_loss_pre.item() + mel_loss_post.mean().item()))
                history["dur_loss"].append(float(dur_loss.item()))
                history["pitch_loss"].append(float(pitch_loss.item()))
                history["energy_loss"].append(float(energy_loss.item()))

        # Validation at end of epoch
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_loader:
                (
                    _,
                    _,
                    _,
                    v_texts,
                    v_src_lens,
                    v_max_src_len,
                    v_mels,
                    v_mel_lens,
                    v_max_mel_len,
                    v_pitches,
                    v_energies,
                    v_durations,
                ) = [x.to(device) if isinstance(x, torch.Tensor) else x for x in val_batch]

                v_preds = model(
                    v_texts,
                    v_src_lens,
                    max_src_len=v_max_src_len,
                    p_targets=v_pitches,
                    e_targets=v_energies,
                    d_targets=v_durations,
                )
                v_targets = (v_texts, v_mels, v_pitches, v_energies, v_durations, v_mels)
                v_loss, _, _, _, _, _ = loss_fn(v_preds, v_targets)
                val_losses.append(v_loss.item())

        mean_val_loss = float(np.mean(val_losses))
        history["val_step"].append(step)
        history["val_loss"].append(mean_val_loss)
        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {np.mean(epoch_losses):.4f} | Val Loss: {mean_val_loss:.4f} | Step: {step}")

        # Checkpoint saving
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(
                {
                    "step": step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "stats": stats,
                    "config": cfg,
                },
                ckpt_dir / "best_model.pth",
            )

    # Save final checkpoint
    torch.save(
        {
            "step": step,
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "val_loss": mean_val_loss,
            "stats": stats,
            "config": cfg,
        },
        ckpt_dir / "checkpoint_20000.pth",
    )
    print(f"Saved checkpoints in {ckpt_dir}")

    # Generate loss plots
    if save_plots:
        plot_loss_curves(history, results_dir, lang)

    return history


def plot_loss_curves(history, results_dir, lang="English"):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training loss plot
    plt.figure(figsize=(9, 5))
    plt.plot(history["step"], history["train_loss"], label="Total Train Loss", color="#1f77b4", lw=2)
    plt.plot(history["step"], history["mel_loss"], label="Mel Loss", color="#ff7f0e", alpha=0.8, lw=1.5)
    plt.plot(history["step"], history["dur_loss"], label="Duration Loss", color="#2ca02c", alpha=0.8, lw=1.5)
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"{lang} FastSpeech 2: Training Loss Convergence", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    train_png = results_dir / "training_loss.png"
    plt.savefig(train_png, dpi=200)
    plt.close()
    print(f"Saved {train_png}")

    # 2. Validation loss plot
    plt.figure(figsize=(9, 5))
    plt.plot(history["val_step"], history["val_loss"], label="Validation Loss", color="#d62728", lw=2.5, marker="o")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.title(f"{lang} FastSpeech 2: Validation Loss Trajectory", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    val_png = results_dir / "validation_loss.png"
    plt.savefig(val_png, dpi=200)
    plt.close()
    print(f"Saved {val_png}")


def main():
    parser = argparse.ArgumentParser(description="Train FastSpeech 2 for English")
    parser.add_argument("--config", default="English/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    args = parser.parse_args()
    train(args.config, num_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
