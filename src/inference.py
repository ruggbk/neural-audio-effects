from pathlib import Path

import soundfile as sf
import torch
import yaml

from model import TCN


def load_model(checkpoint_path: Path, config: dict, device: torch.device = torch.device("cpu")) -> TCN:
    model = TCN(
        channels=config.get("channels", 32),
        n_layers=config.get("n_layers", 10),
        n_stacks=config.get("n_stacks", 2),
    ).to(device)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    config = yaml.safe_load(open(repo_root / args.config))
    checkpoint_path = repo_root / "models" / f"{config['name']}.pt"

    input_path = Path(args.input) if args.input else (
        repo_root / "data" / "guitarset" / "audio" / "00_BN1-129-Eb_comp_mix.wav"
    )
    output_path = Path(args.output) if args.output else (
        repo_root / "samples" / f"00_BN1-129-Eb_comp_predicted_{config['name']}.wav"
    )

    model = load_model(checkpoint_path, config)
    run_inference(model=model, input_path=input_path, output_path=output_path)
    print(f"Done. Output: {output_path}")
