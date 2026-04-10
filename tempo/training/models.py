from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def lengths_to_padding_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError("Expected lengths to have shape [batch].")
    max_length = max_length or int(lengths.max().item())
    time_index = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return time_index >= lengths.unsqueeze(1)


def _hz_to_mel(frequencies_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + frequencies_hz / 700.0)


def _mel_to_hz(mel_values: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel_values / 2595.0) - 1.0)


def build_mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> torch.Tensor:
    max_frequency = float(f_max if f_max is not None else sample_rate / 2.0)
    fft_frequency_bins = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    mel_points = torch.linspace(
        _hz_to_mel(torch.tensor(f_min)),
        _hz_to_mel(torch.tensor(max_frequency)),
        n_mels + 2,
    )
    hz_points = _mel_to_hz(mel_points)

    filterbank = torch.zeros(n_mels, n_fft // 2 + 1)
    for band in range(n_mels):
        left = float(hz_points[band].item())
        center = float(hz_points[band + 1].item())
        right = float(hz_points[band + 2].item())

        left_denom = max(center - left, 1e-6)
        right_denom = max(right - center, 1e-6)
        left_slope = (fft_frequency_bins - left) / left_denom
        right_slope = (right - fft_frequency_bins) / right_denom
        filterbank[band] = torch.minimum(left_slope, right_slope).clamp_min(0.0)

    return filterbank


def build_activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "mish":
        return nn.Mish()
    raise ValueError(f"Unsupported activation: {name}")


@dataclass(frozen=True)
class StreamingModelConfig:
    model_type: str = "conformer"
    sample_rate: int = 16_000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float | None = 7_600.0
    encoder_dim: int = 144
    output_dim: int = 128
    dropout: float = 0.1
    activation: str = "silu"
    causal: bool = True
    conformer_layers: int = 4
    conformer_heads: int = 4
    conformer_ffn_multiplier: int = 4
    conformer_conv_kernel_size: int = 31
    rnnt_layers: int = 4
    rnnt_time_reduction_factor: int = 2


class LogMelFrontend(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float | None,
        log_floor: float = 1e-5,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.log_floor = log_floor
        self.center = center

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer(
            "mel_filterbank",
            build_mel_filterbank(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
            ),
            persistent=False,
        )

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if waveforms.ndim != 2:
            raise ValueError("Expected waveforms to have shape [batch, samples].")

        stft = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )
        power = stft.abs().pow(2.0)
        features = torch.einsum("mf,bft->btm", self.mel_filterbank, power)
        features = torch.log(features.clamp_min(self.log_floor))

        feature_mean = features.mean(dim=1, keepdim=True)
        feature_std = features.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        features = (features - feature_mean) / feature_std

        feature_lengths = self.output_lengths(lengths).clamp(max=features.size(1), min=1)
        return features, feature_lengths

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        if self.center:
            return 1 + torch.div(lengths, self.hop_length, rounding_mode="floor")
        valid = (lengths - self.win_length).clamp_min(0)
        return 1 + torch.div(valid, self.hop_length, rounding_mode="floor")


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        time_steps = sequence.size(1)
        positions = torch.arange(time_steps, device=sequence.device, dtype=sequence.dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=sequence.device, dtype=sequence.dtype)
            * (-math.log(10_000.0) / self.dim)
        )
        encoding = torch.zeros(time_steps, self.dim, device=sequence.device, dtype=sequence.dtype)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        return self.dropout(sequence + encoding.unsqueeze(0))


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, multiplier: int, dropout: float, activation: str) -> None:
        super().__init__()
        hidden_dim = dim * multiplier
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(sequence))


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            groups=groups,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence = F.pad(sequence, (self.left_padding, 0))
        return self.conv(sequence)


class ConformerConvolutionModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float, causal: bool, activation: str) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.depthwise = (
            CausalConv1d(dim, dim, kernel_size=kernel_size, groups=dim)
            if causal
            else nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim,
            )
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = build_activation(activation)
        self.pointwise_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(sequence).transpose(1, 2)
        hidden = F.glu(self.pointwise_in(hidden), dim=1)
        hidden = self.depthwise(hidden)
        hidden = self.batch_norm(hidden)
        hidden = self.activation(hidden)
        hidden = self.pointwise_out(hidden)
        hidden = self.dropout(hidden)
        return hidden.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        ffn_multiplier: int,
        conv_kernel_size: int,
        dropout: float,
        causal: bool,
        activation: str,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.ffn1 = FeedForwardModule(dim, ffn_multiplier, dropout, activation)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvolutionModule(dim, conv_kernel_size, dropout, causal, activation)
        self.ffn2 = FeedForwardModule(dim, ffn_multiplier, dropout, activation)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, sequence: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        sequence = sequence + 0.5 * self.ffn1(sequence)

        attn_mask = None
        if self.causal:
            time_steps = sequence.size(1)
            attn_mask = torch.triu(
                torch.ones(time_steps, time_steps, device=sequence.device, dtype=torch.bool),
                diagonal=1,
            )

        attn_input = self.attn_norm(sequence)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        sequence = sequence + self.attn_dropout(attn_output)
        sequence = sequence + self.conv(sequence)
        sequence = sequence + 0.5 * self.ffn2(sequence)
        sequence = self.final_norm(sequence)
        return sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class ConformerEncoder(nn.Module):
    def __init__(self, config: StreamingModelConfig) -> None:
        super().__init__()
        if config.encoder_dim % config.conformer_heads != 0:
            raise ValueError("encoder_dim must be divisible by conformer_heads.")

        self.input_norm = nn.LayerNorm(config.n_mels)
        self.input_projection = nn.Linear(config.n_mels, config.encoder_dim)
        self.input_activation = build_activation(config.activation)
        self.positional_encoding = SinusoidalPositionalEncoding(config.encoder_dim, config.dropout)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    config.encoder_dim,
                    num_heads=config.conformer_heads,
                    ffn_multiplier=config.conformer_ffn_multiplier,
                    conv_kernel_size=config.conformer_conv_kernel_size,
                    dropout=config.dropout,
                    causal=config.causal,
                    activation=config.activation,
                )
                for _ in range(config.conformer_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.encoder_dim)
        self.output_projection = nn.Linear(config.encoder_dim, config.output_dim)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = lengths_to_padding_mask(lengths, max_length=features.size(1))
        hidden = self.input_activation(self.input_projection(self.input_norm(features)))
        hidden = self.positional_encoding(hidden)

        for layer in self.layers:
            hidden = layer(hidden, padding_mask)

        hidden = self.output_projection(self.output_norm(hidden))
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return hidden, lengths


class RNNTStyleEncoder(nn.Module):
    def __init__(self, config: StreamingModelConfig) -> None:
        super().__init__()
        self.time_reduction_factor = max(1, config.rnnt_time_reduction_factor)
        input_dim = config.n_mels * self.time_reduction_factor
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, config.encoder_dim)
        self.input_activation = build_activation(config.activation)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=config.encoder_dim,
                    hidden_size=config.encoder_dim,
                    num_layers=1,
                    batch_first=True,
                )
                for _ in range(config.rnnt_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.encoder_dim)
        self.output_projection = nn.Linear(config.encoder_dim, config.output_dim)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, lengths = self._stack_time(features, lengths)
        hidden = self.input_activation(self.input_projection(self.input_norm(features)))

        for layer in self.layers:
            packed = pack_padded_sequence(
                hidden,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = layer(packed)
            hidden_unpacked, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=hidden.size(1),
            )
            hidden = self.dropout(hidden + hidden_unpacked)

        padding_mask = lengths_to_padding_mask(lengths, max_length=hidden.size(1))
        hidden = self.output_projection(self.output_norm(hidden))
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return hidden, lengths

    def _stack_time(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.time_reduction_factor == 1:
            return features, lengths

        batch_size, time_steps, feature_dim = features.shape
        factor = self.time_reduction_factor
        pad_steps = (-time_steps) % factor
        if pad_steps:
            features = F.pad(features, (0, 0, 0, pad_steps))

        reduced = features.view(batch_size, -1, feature_dim * factor)
        reduced_lengths = torch.div(
            lengths + factor - 1,
            factor,
            rounding_mode="floor",
        ).clamp_min(1)
        return reduced, reduced_lengths


class StreamingEmotionModel(nn.Module):
    def __init__(self, config: StreamingModelConfig) -> None:
        super().__init__()
        self.config = config
        self.frontend = LogMelFrontend(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
        )
        if config.model_type == "conformer":
            self.encoder = ConformerEncoder(config)
        elif config.model_type == "rnnt":
            self.encoder = RNNTStyleEncoder(config)
        else:
            raise ValueError(f"Unsupported model_type: {config.model_type}")

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, feature_lengths = self.frontend(waveforms, lengths)
        return self.encoder(features, feature_lengths)


def build_streaming_emotion_model(config: StreamingModelConfig) -> StreamingEmotionModel:
    return StreamingEmotionModel(config)
