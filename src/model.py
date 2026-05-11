import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        # Left-only padding keeps the convolution causal (no future samples)
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(F.pad(x, (self.left_pad, 0)))
        return x + self.activation(out)


class TCN(nn.Module):
    """Temporal Convolutional Network for audio effects modeling.

    Stacks dilated causal conv blocks with doubling dilation (1, 2, 4, ..., 512)
    repeated across multiple stacks. Each block has a residual connection.
    Input and output are mono waveforms of shape (batch, 1, samples).
    """

    def __init__(
        self,
        channels: int = 32,
        kernel_size: int = 3,
        n_layers: int = 10,
        n_stacks: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(1, channels, 1)

        dilations = [2**i for i in range(n_layers)] * n_stacks
        self.blocks = nn.ModuleList([
            TCNBlock(channels, kernel_size, d) for d in dilations
        ])

        self.output_proj = nn.Conv1d(channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)

    def receptive_field(self) -> int:
        """Number of input samples each output sample depends on."""
        return sum(block.left_pad for block in self.blocks) + 1
