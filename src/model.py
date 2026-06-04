import torch
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


class _CausalConv1d(nn.Module):
    """Conv1d with left-only padding — causal at any stride or dilation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class WaveUNet(nn.Module):
    """Causal Wave-U-Net for audio timbre transfer.

    Encoder: stride-2 causal Conv1d halves time resolution at each of `depth` levels.
    Bottleneck: dilated TCN blocks on the coarsest feature map.
    Decoder: nearest-neighbour upsample → skip concat → causal Conv1d, bottom-to-top.
    The finest skip is the raw input (1 channel); each coarser skip is the encoder output
    at that resolution. All convolutions use left-only padding, so algorithmic latency is zero.
    Output is trimmed/padded to exactly match input length.

    Input/output shape: (batch, 1, samples).

    Channel progression (enc_ch): [1, C, 2C, 4C, ..., 2^(depth-1)*C]
    Decoder stage j (0 = deepest):
        in  = enc_ch[depth-j]  (upsampled) + enc_ch[depth-1-j]  (skip)
        out = enc_ch[depth-1-j]  (j < depth-1),  else C  (final stage)
    """

    def __init__(
        self,
        channels: int = 16,
        depth: int = 3,
        kernel_size: int = 15,
        bottleneck_layers: int = 8,
        window_size: int = 4096,
    ):
        super().__init__()

        # [1, C, 2C, 4C, ...] — enc_ch[i+1] is the output channel count of enc_blocks[i]
        enc_ch = [1] + [channels * (2 ** i) for i in range(depth)]

        self.enc_blocks = nn.ModuleList([
            nn.Sequential(
                _CausalConv1d(enc_ch[i], enc_ch[i + 1], kernel_size, stride=2),
                nn.GELU(),
            )
            for i in range(depth)
        ])

        # Bottleneck: dilated TCN on enc_ch[-1]-channel features.
        # Dilation pattern repeats at a period of floor(log2(window_size / 2^depth)) so
        # no layer samples beyond the bottleneck window into zero-padding.
        # e.g. window_size=4096, depth=3 → bottleneck=512 → period=8 → max dilation=128.
        bn_ch = enc_ch[-1]
        bottleneck_len = window_size >> depth
        period = max(1, bottleneck_len.bit_length() - 2)
        dilations = [2 ** (i % period) for i in range(bottleneck_layers)]
        self.bottleneck = nn.ModuleList([
            TCNBlock(bn_ch, kernel_size=3, dilation=d) for d in dilations
        ])

        # Decoder conv for each stage j:
        #   j=0 (deepest): in = enc_ch[depth] + enc_ch[depth-1], out = enc_ch[depth-1]
        #   j=k:           in = enc_ch[depth-k] + enc_ch[depth-1-k], out = enc_ch[depth-1-k]
        #   j=depth-1:     in = enc_ch[1] + enc_ch[0] = C+1,         out = C (channels)
        self.dec_convs = nn.ModuleList()
        for j in range(depth):
            in_ch = enc_ch[depth - j] + enc_ch[depth - 1 - j]
            out_ch = channels if j == depth - 1 else enc_ch[depth - 1 - j]
            self.dec_convs.append(
                nn.Sequential(
                    _CausalConv1d(in_ch, out_ch, kernel_size),
                    nn.GELU(),
                )
            )

        self.output_proj = nn.Conv1d(channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        depth = len(self.enc_blocks)

        # Encoder
        skips = []
        h = x
        for block in self.enc_blocks:
            h = block(h)
            skips.append(h)

        # Bottleneck
        for block in self.bottleneck:
            h = block(h)

        # Decoder: upsample → trim to skip length → cat → conv
        for j, dec_conv in enumerate(self.dec_convs):
            h = F.interpolate(h, scale_factor=2.0, mode='nearest')

            # skips[-1] is the deepest encoder output (= bottleneck input), not used as a skip.
            # skips[depth-2-j] is the encoder output at the matching resolution.
            # The final stage uses the original input as the 1-channel finest-resolution skip.
            skip = x if j == depth - 1 else skips[depth - 2 - j]

            # Trim to the shorter length; stride rounding can cause a ±1 sample mismatch.
            L = min(h.shape[-1], skip.shape[-1])
            h = torch.cat([h[..., :L], skip[..., :L]], dim=1)
            h = dec_conv(h)

        out = self.output_proj(h)
        # Restore exact input length (strided rounding may shift output by a few samples).
        L_in = x.shape[-1]
        if out.shape[-1] < L_in:
            out = F.pad(out, (0, L_in - out.shape[-1]))
        else:
            out = out[..., :L_in]
        return out
