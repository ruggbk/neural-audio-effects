from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from dataset import make_datasets
from model import TCN


BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100
# num_workers > 0 can cause issues on Windows; 0 is safe
NUM_WORKERS = 0


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-scale magnitude spectrogram loss across three FFT sizes."""
    losses = []
    for n_fft in [512, 1024, 2048]:
        window = torch.hann_window(n_fft, device=pred.device)
        pred_mag = torch.stft(pred.squeeze(1), n_fft=n_fft, window=window, return_complex=True).abs()
        target_mag = torch.stft(target.squeeze(1), n_fft=n_fft, window=window, return_complex=True).abs()
        losses.append(F.l1_loss(pred_mag, target_mag))
    return sum(losses)


def train(config_path: Path, repo_root: Path, resume_from: Optional[Path] = None) -> None:
    """Train the TCN model.

    Args:
        config_path: Path to the experiment YAML config.
        repo_root: Root of the repository.
        resume_from: Optional path to a checkpoint to resume training from.
    """
    config = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds = make_datasets(
        rendered_dir=repo_root / config["output_dir"],
        guitar_dir=repo_root / "data" / "guitarset" / "audio",
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    print(f"Train: {len(train_ds):,} windows | Val: {len(val_ds):,} windows")

    model = TCN().to(device)
    if resume_from is not None:
        model.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"Resumed from {resume_from}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    checkpoint_path = repo_root / "models" / f"{config['name']}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for guitar, rendered in train_loader:
            guitar, rendered = guitar.to(device), rendered.to(device)
            pred = model(guitar)
            loss = F.mse_loss(pred, rendered) + spectral_loss(pred, rendered)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for guitar, rendered in val_loader:
                guitar, rendered = guitar.to(device), rendered.to(device)
                pred = model(guitar)
                val_loss += (F.mse_loss(pred, rendered) + spectral_loss(pred, rendered)).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"           -> saved checkpoint")

    print(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    train(
        config_path=repo_root / "configs" / "b3_organ.yaml",
        repo_root=repo_root,
        # resume_from=repo_root / "models" / "b3_organ_smoke.pt",
    )
