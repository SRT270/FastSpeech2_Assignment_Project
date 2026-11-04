"""
FastSpeech 2 Complete Neural Network Architecture:
Transformer Encoder, Variance Adaptor, Transformer Decoder, Mel Projection, and Post-Net.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_mask_from_lengths
from .variance_adaptor import VarianceAdaptor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_head=2, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v)
        self.fc = nn.Linear(n_head * self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        residual = q
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask: (batch, 1, 1, len_k) or (batch, 1, len_q, len_k)
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output


class PositionwiseFeedForward(nn.Module):
    """Feed-Forward module using 1D convolutions."""

    def __init__(self, d_in=256, d_hid=1024, kernel_size=9, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=1,
            padding=0,
        )
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class FFTBlock(nn.Module):
    """Feed-Forward Transformer Block."""

    def __init__(self, d_model=256, n_head=2, d_hid=1024, kernel_size=9, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_hid, kernel_size=kernel_size, dropout=dropout
        )

    def forward(self, enc_input, mask=None):
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, mask=mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class Encoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        d_model=256,
        n_layer=4,
        n_head=2,
        d_hid=1024,
        kernel_size=9,
        dropout=0.1,
    ):
        super(Encoder, self).__init__()
        self.src_word_emb = nn.Embedding(n_vocab, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(d_model, n_head, d_hid, kernel_size, dropout)
                for _ in range(n_layer)
            ]
        )

    def forward(self, src_seq, src_mask):
        enc_output = self.src_word_emb(src_seq)
        enc_output = self.pos_enc(enc_output)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, mask=src_mask)
        return enc_output


class Decoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_layer=6,
        n_head=2,
        d_hid=1024,
        kernel_size=9,
        dropout=0.1,
    ):
        super(Decoder, self).__init__()
        self.pos_enc = PositionalEncoding(d_model)
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(d_model, n_head, d_hid, kernel_size, dropout)
                for _ in range(n_layer)
            ]
        )

    def forward(self, dec_seq, dec_mask):
        dec_output = self.pos_enc(dec_seq)
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, mask=dec_mask)
        return dec_output


class PostNet(nn.Module):
    """5-layer 1D Conv Post-Net to refine mel spectrograms."""

    def __init__(self, n_mel_channels=80, postnet_embedding_dim=512, postnet_kernel_size=5):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=(postnet_kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
                nn.Tanh(),
                nn.Dropout(0.5),
            )
        )

        for _ in range(3):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=(postnet_kernel_size - 1) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                    nn.Tanh(),
                    nn.Dropout(0.5),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=(postnet_kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.Dropout(0.5),
            )
        )

    def forward(self, x):
        # x: (B, T, n_mels) -> transpose to (B, n_mels, T)
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = conv(x)
        return x.transpose(1, 2)


class FastSpeech2(nn.Module):
    """FastSpeech 2 End-to-End Non-Autoregressive Text-to-Speech Model."""

    def __init__(
        self,
        n_vocab=150,
        n_mel_channels=80,
        encoder_layer=4,
        encoder_head=2,
        encoder_hidden=256,
        decoder_layer=6,
        decoder_head=2,
        decoder_hidden=256,
        fft_conv_filter_size=1024,
        fft_conv_kernel_size=9,
        encoder_dropout=0.1,
        decoder_dropout=0.1,
        variance_filter_size=256,
        variance_kernel_size=3,
        variance_dropout=0.5,
        n_bins=256,
        pitch_min=60.0,
        pitch_max=450.0,
        energy_min=0.0,
        energy_max=100.0,
    ):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(
            n_vocab=n_vocab,
            d_model=encoder_hidden,
            n_layer=encoder_layer,
            n_head=encoder_head,
            d_hid=fft_conv_filter_size,
            kernel_size=fft_conv_kernel_size,
            dropout=encoder_dropout,
        )

        self.variance_adaptor = VarianceAdaptor(
            hidden_dim=encoder_hidden,
            filter_size=variance_filter_size,
            kernel_size=variance_kernel_size,
            dropout=variance_dropout,
            n_bins=n_bins,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            energy_min=energy_min,
            energy_max=energy_max,
        )

        self.decoder = Decoder(
            d_model=decoder_hidden,
            n_layer=decoder_layer,
            n_head=decoder_head,
            d_hid=fft_conv_filter_size,
            kernel_size=fft_conv_kernel_size,
            dropout=decoder_dropout,
        )

        self.mel_linear = nn.Linear(decoder_hidden, n_mel_channels)
        self.postnet = PostNet(n_mel_channels=n_mel_channels)

    def forward(
        self,
        texts,
        src_lens,
        max_src_len=None,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_mask = get_mask_from_lengths(src_lens, max_src_len)

        output = self.encoder(texts, src_mask)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_mask,
        ) = self.variance_adaptor(
            output,
            src_mask,
            max_mel_len=max_mel_len,
            p_target=p_targets,
            e_target=e_targets,
            d_target=d_targets,
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
        )

        output = self.decoder(output, mel_mask)
        mel_before_postnet = self.mel_linear(output)

        postnet_output = self.postnet(mel_before_postnet)
        mel_after_postnet = mel_before_postnet + postnet_output

        return (
            mel_before_postnet,
            mel_after_postnet,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_mask,
            mel_mask,
            mel_lens,
        )

