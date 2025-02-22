#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MixIT training on WSJ0-2Mix (8 kHz min) using a TDCN++ architecture
that closely matches the TensorFlow snippet from `improved_tdcn`.
-------------------------------------------------------------------
Usage:
  python train_mixit_tdcnpp.py \
      --train_dir /path/to/wsj0-mix/2speakers/wav8k/min/tr/mix \
      --valid_dir /path/to/wsj0-mix/2speakers/wav8k/min/cv/mix \
      --test_dir  /path/to/wsj0-mix/2speakers/wav8k/min/tt/mix \
      --epochs 20 \
      --batch_size 4 \
      --lr 1e-3 \
      --project_name MyMixITProject
"""

import argparse
import glob
import os
import random

import numpy as np
import torch
import torchaudio
from asteroid.losses import MixITLossWrapper, multisrc_neg_sisdr

# Asteroid library for speech separation (with TDCN++ support):
# pip install git+https://github.com/asteroid-team/asteroid
from asteroid.models import ConvTasNet
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb


# ----------------------
# 1. Dataset Definition
# ----------------------
class Wsj0MixPairedDataset(Dataset):
    """
    This Dataset:
      - Reads a directory of 2-speaker mixture WAV files (WSJ0-2Mix style).
      - On __getitem__, returns a *pair* of mixture waveforms (mix1, mix2)
        that do NOT share any speaker ID, ensuring disjoint sets of speakers.
      - We fix the dataset length as num_items, sampling random pairs each time.
    """

    def __init__(self, mix_dir, sample_rate=8000, max_len=4.0, num_items=10000):
        """
        Args:
            mix_dir (str): Path to folder containing 2-speaker mixture WAV files.
            sample_rate (int): 8kHz for WSJ0-2Mix.
            max_len (float): We'll random-crop or zero-pad mixtures to this length (in seconds).
            num_items (int): "Length" of dataset for PyTorch. Each item is random.
        """
        super().__init__()
        self.mix_dir = mix_dir
        self.sample_rate = sample_rate
        self.num_items = num_items

        # Collect mixture files
        self.wav_paths = sorted(glob.glob(os.path.join(mix_dir, "*.wav")))
        if len(self.wav_paths) == 0:
            raise RuntimeError(f"No .wav files found in {mix_dir}.")

        # Parse speaker IDs from each mixture's filename
        self.data = []
        for wp in self.wav_paths:
            fname = os.path.basename(
                wp
            )  # e.g. 011a0101_0.061105_401c020r_-0.061105.wav
            splitted = fname.replace(".wav", "").split("_")
            if len(splitted) < 3:
                # Unexpected naming => skip
                continue
            spk_id_1 = splitted[0]
            spk_id_2 = splitted[2]
            spk_set = set([spk_id_1, spk_id_2])
            self.data.append((wp, spk_set))

        self.max_samples = int(sample_rate * max_len)

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        """
        Returns (mix1, mix2) waveforms [both shape (time,)],
        such that they do NOT share any speaker IDs.
        """
        while True:
            wp1, spk_set_1 = random.choice(self.data)
            wp2, spk_set_2 = random.choice(self.data)
            if spk_set_1.isdisjoint(spk_set_2):
                break

        # Load waveforms => shape (channel, time)
        mix1, sr1 = torchaudio.load(wp1)
        mix2, sr2 = torchaudio.load(wp2)
        assert sr1 == sr2 == self.sample_rate, "Sample rate mismatch"

        # Convert to shape (time,)
        mix1 = mix1[0]
        mix2 = mix2[0]

        # Crop/Pad to self.max_samples
        mix1 = self._fix_length(mix1)
        mix2 = self._fix_length(mix2)

        return mix1, mix2

    def _fix_length(self, wav):
        """
        If wav is longer than max_samples, random-crop. If shorter, zero-pad.
        """
        length = wav.shape[0]
        if length > self.max_samples:
            start = random.randint(0, length - self.max_samples - 1)
            wav = wav[start : start + self.max_samples]
        elif length < self.max_samples:
            pad_len = self.max_samples - length
            wav = torch.cat([wav, torch.zeros(pad_len)], dim=0)
        return wav


def collate_fn(batch):
    """
    Collates a list of (mix1, mix2) => two tensors of shape (B, time).
    """
    mix1_list, mix2_list = [], []
    for m1, m2 in batch:
        mix1_list.append(m1.unsqueeze(0))
        mix2_list.append(m2.unsqueeze(0))
    mix1 = torch.cat(mix1_list, dim=0)  # (B, time)
    mix2 = torch.cat(mix2_list, dim=0)  # (B, time)
    return mix1, mix2


# ----------------------
# 2. TDCN++ Module (Asteroid)
# ----------------------
class TDCNppMixIT(nn.Module):
    """
    Wraps a TDCN++ network from Asteroid, configured to closely match the
    'improved_tdcn' snippet's hyperparameters.
    """

    def __init__(self, n_src=2, sample_rate=8000):
        super().__init__()

        # TDCN++ config that approximates the snippet:
        #  - bottleneck=256  => bn_chan=256
        #  - conv_channels=512 => hidden_chan=512
        #  - kernel_size=3
        #  - num_dilations=8, num_repeats=4 => n_blocks=32, n_repeats=4
        #  - norm_type='instance_norm' => approximated here with 'gLN' (global layer norm)
        #  - activation='prelu'
        #
        # Note: This won't replicate extra skip connections or exponential scaling
        # (which the snippet can do).
        self.model = ConvTasNet(
            n_src=n_src,
            sample_rate=sample_rate,
            masknet="tdcn++",
            masknet_config={
                "bn_chan": 256,
                "hidden_chan": 512,
                "skip_chan": 256,
                "n_blocks": 32,  # 8 dilations * 4 repeats
                "n_repeats": 4,
                "kernel_size": 3,
                "norm_type": "gLN",  # or "cLN" or "batchnorm"
                "activation": "prelu",
                "use_skip": True,
                # If you want norm after activation:
                "norm_after_act": False,
            },
            # Potentially set the encoder kernel size if you want.
            # The snippet doesn't specify it exactly. 16 is typical for 8kHz.
            encoder_kernel_size=16,
            encoder_activation="relu",
        )

        # MixIT loss: negative SISDR
        self.mixit_loss_func = MixITLossWrapper(multisrc_neg_sisdr, generalized=True)

    def forward(self, mixture_of_mixtures):
        # mixture_of_mixtures => shape (B, time)
        # TDCN++ returns => shape (B, n_src, time)
        return self.model(mixture_of_mixtures)

    def mixit_loss(self, est_sources, references):
        # references => shape (B, n_references, time)
        # est_sources => shape (B, n_src, time)
        # returns shape (B,) => per-sample loss
        return self.mixit_loss_func(est_sources, references)


# ----------------------
# 3. Train + Evaluate
# ----------------------
def train_mixit_tdcnpp(
    train_dir,
    valid_dir,
    test_dir,
    epochs=10,
    batch_size=4,
    lr=1e-3,
    sr=8000,
    max_len=4,
    num_workers=1,
    train_num_items=5,
    valid_num_items=5,
    test_num_items=5,
    project_name="MixIT-TDCN++",
):

    # ------------------ WandB Init ------------------
    wandb.init(project=project_name)
    wandb.config.update(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "sample_rate": sr,
        }
    )

    # Create Datasets + DataLoaders
    train_ds = Wsj0MixPairedDataset(
        mix_dir=train_dir,
        sample_rate=sr,
        max_len=max_len,  # e.g. 4-second random segments
        num_items=train_num_items,
    )
    valid_ds = Wsj0MixPairedDataset(
        mix_dir=valid_dir,
        sample_rate=sr,
        max_len=max_len,
        num_items=valid_num_items,
    )
    test_ds = Wsj0MixPairedDataset(
        mix_dir=test_dir,
        sample_rate=sr,
        max_len=max_len,
        num_items=test_num_items,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,  # random anyway
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,
    )

    # Model & Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TDCNppMixIT(n_src=2, sample_rate=sr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        # Wrap train_loader with tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        for mix1, mix2 in pbar:
            mix1 = mix1.to(device)  # (B, T)
            mix2 = mix2.to(device)

            # Mixture of mixtures
            mom = mix1 + mix2  # (B, T)

            # Forward pass => est_sources => (B, n_src, T)
            est_sources = model(mom)

            # References => (B, 2, T)
            references = torch.stack([mix1, mix2], dim=1)

            # MixIT loss => shape (B,)
            loss_per_sample = model.mixit_loss(est_sources, references)
            loss = loss_per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            global_step += 1

            # Optionally update tqdm display
            pbar.set_postfix({"loss": loss.item()})

            # Log every 50 steps to wandb
            if global_step % 50 == 0:
                wandb.log({"train/loss_step": loss.item(), "global_step": global_step})

        avg_train_loss = float(np.mean(train_losses))
        wandb.log({"train/loss_epoch": avg_train_loss, "epoch": epoch})

        # Validation
        val_loss = evaluate(model, valid_loader, device)
        wandb.log({"val/loss_epoch": val_loss, "epoch": epoch})
        print(
            f"[Epoch {epoch}] train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}"
        )
    # Test
    test_loss = evaluate(model, test_loader, device)
    wandb.log({"test/loss_final": test_loss})
    print(f"Final Test Loss (MixIT TDCN++): {test_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), "mixit_tdcnpp_model.pt")
    print("Saved TDCN++ MixIT model => mixit_tdcnpp_model.pt")


def evaluate(model: TDCNppMixIT, loader: DataLoader, device: torch.device):
    model.eval()
    losses = []
    with torch.no_grad():
        for mix1, mix2 in loader:
            mix1 = mix1.to(device)
            mix2 = mix2.to(device)
            mom = mix1 + mix2
            est_sources = model(mom)
            references = torch.stack([mix1, mix2], dim=1)
            loss_per_sample = model.mixit_loss(est_sources, references)
            losses.append(loss_per_sample.mean().item())
    return float(np.mean(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir",
        type=str,
        default=r"C:\Users\arifr\git\mixit\wsj0-mix\2speakers\wav8k\max\tr\mix",
        help="Path to 'tr/mix' folder of WSJ0-2Mix 8kHz min",
    )
    parser.add_argument(
        "--valid_dir",
        type=str,
        default=r"C:\Users\arifr\git\mixit\wsj0-mix\2speakers\wav8k\max\cv\mix",
        help="Path to 'cv/mix' folder",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=r"C:\Users\arifr\git\mixit\wsj0-mix\2speakers\wav8k\max\tt\mix",
        help="Path to 'tt/mix' folder",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--project_name", type=str, default="MixIT-TDCN++")
    parser.add_argument("--max_len", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--train_num_items", type=int, default=5)
    parser.add_argument("--valid_num_items", type=int, default=5)
    parser.add_argument("--test_num_items", type=int, default=5)

    args = parser.parse_args()
    train_mixit_tdcnpp(
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        test_dir=args.test_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sr=args.sr,
        num_workers=args.num_workers,
        train_num_items=args.train_num_items,
        valid_num_items=args.valid_num_items,
        test_num_items=args.test_num_items,
        project_name=args.project_name,
    )
