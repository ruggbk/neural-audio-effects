import argparse
from pathlib import Path
from typing import Dict, List, Union

import torch
import yaml
from torch import Tensor
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model

from model import TCN


class TCNWrapper(WaveformToWaveformBase):
    # TorchScript does not inherit class attribute type annotations from parent classes.
    # Redeclare SDK attributes and our own typed attributes here as a workaround.
    # https://github.com/pytorch/pytorch/issues/51041#issuecomment-767061194
    neutone_parameters_metadata: Dict[
        str, Dict[str, Union[int, float, str, bool, List[str], List[int]]]
    ]
    remapped_params: Dict[str, Tensor]
    neutone_parameter_names: List[str]
    _meta_name: str
    _meta_version: str
    _meta_short_description: str
    _meta_long_description: str
    _meta_technical_description: str
    _meta_tags: List[str]
    _meta_sample_rate: int

    def __init__(self, model: TCN, meta: dict):
        self._meta_name = meta["display_name"]
        self._meta_version = meta["version"]
        self._meta_short_description = meta["short_description"]
        self._meta_long_description = meta["long_description"]
        self._meta_technical_description = meta["technical_description"]
        self._meta_tags = meta["tags"]
        self._meta_sample_rate = meta["sample_rate"]
        super().__init__(model)

    def get_model_name(self) -> str:
        return self._meta_name

    def get_model_authors(self) -> List[str]:
        return ["Brandon Rugg"]

    def get_model_version(self) -> str:
        return self._meta_version

    def get_model_short_description(self) -> str:
        return self._meta_short_description

    def get_model_long_description(self) -> str:
        return self._meta_long_description

    def get_technical_description(self) -> str:
        return self._meta_technical_description

    def get_tags(self) -> List[str]:
        return self._meta_tags

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return []

    def is_input_mono(self) -> bool:
        return True

    def is_output_mono(self) -> bool:
        return True

    def get_native_sample_rates(self) -> List[int]:
        return [self._meta_sample_rate]

    def get_native_buffer_sizes(self) -> List[int]:
        return []

    def is_experimental(self) -> bool:
        return True

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # x: (1, n_samples) — add batch dim, run model, remove batch dim
        return self.model(x.unsqueeze(0)).squeeze(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    config = yaml.safe_load(open(args.config))

    checkpoint_path = repo_root / "models" / f"{config['name']}.pt"
    output_dir = repo_root / "models" / "neutone_export" / config["name"]

    tcn = TCN()
    tcn.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    wrapper = TCNWrapper(tcn, config)
    save_neutone_model(wrapper, output_dir, dump_samples=False)
    print(f"Exported to {output_dir}")
