from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model

from model import TCN


class B3OrganWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "B3 Organ Effect"

    def get_model_authors(self) -> List[str]:
        return ["Brandon Rugg"]

    def get_model_version(self) -> str:
        return "0.1.0"

    def get_model_short_description(self) -> str:
        return "TCN trained to transform guitar into Hammond B3 organ character."

    def get_model_long_description(self) -> str:
        return (
            "A Temporal Convolutional Network trained on paired guitar and Hammond B3 "
            "organ audio derived from GuitarSet. Input is clean DI guitar; output "
            "approximates the timbral character of a Hammond B3 through a rotary speaker."
        )

    def get_technical_description(self) -> str:
        return (
            "Dilated causal TCN (2 stacks x 10 layers, 32 channels, kernel size 3). "
            "Receptive field: 4093 samples (~93ms at 44100Hz). "
            "Trained with MSE + multi-scale spectral loss on GuitarSet audio paired "
            "with MIDI-driven VSTi renders."
        )

    def get_tags(self) -> List[str]:
        return ["guitar", "organ", "timbral transfer", "TCN"]

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return []

    def is_input_mono(self) -> bool:
        return True

    def is_output_mono(self) -> bool:
        return True

    def get_native_sample_rates(self) -> List[int]:
        return [44100]

    def get_native_buffer_sizes(self) -> List[int]:
        return []

    def is_experimental(self) -> bool:
        return True

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # x: (1, n_samples) — add batch dim, run model, remove batch dim
        return self.model(x.unsqueeze(0)).squeeze(0)


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    checkpoint_path = repo_root / "models" / "b3_organ_smoke.pt"
    output_dir = repo_root / "models" / "neutone_export"

    tcn = TCN()
    tcn.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    wrapper = B3OrganWrapper(tcn)
    save_neutone_model(wrapper, output_dir, dump_samples=False)
    print(f"Exported to {output_dir}")
