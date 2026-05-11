from pathlib import Path

import soundfile as sf
import torch

from model import TCN


def load_model(checkpoint_path: Path, device: torch.device = torch.device("cpu")) -> TCN:
    """Load a TCN model from a checkpoint file."""
    model = TCN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def run_inference(model: TCN, input_path: Path, output_path: Path) -> None:
    """Run the model on a full audio file and write the output to disk.

    Processes the entire file in one pass — suitable for offline rendering.
    Input is converted to mono if stereo.
    """
    audio, sr = sf.read(str(input_path), always_2d=True)
    audio_t = torch.from_numpy(audio.T[:1]).float().unsqueeze(0)

    with torch.no_grad():
        output = model(audio_t)

    output_np = output.squeeze().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), output_np, sr)


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    model = load_model(repo_root / "models" / "b3_organ_smoke.pt")
    run_inference(
        model=model,
        input_path=repo_root / "data" / "guitarset" / "audio" / "00_BN1-129-Eb_comp_mix.wav",
        output_path=repo_root / "samples" / "00_BN1-129-Eb_comp_predicted.wav",
    )
    print("Done.")
