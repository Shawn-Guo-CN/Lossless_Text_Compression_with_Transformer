"""
Definition of modules in GPT and Mistral language models.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) Huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/
transformers/models/gpt2/modeling_gpt2.py
3) the official Mistral implementation by MistralAI:
https://github.com/mistralai/mistral-src/
"""
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import Configs


@dataclass
class MoeArgs(Configs):
    num_experts:         int
    num_experts_per_tok: int


@dataclass
class ModelArgs(Configs):
    model_type: Union[str, None]
    cxt_size:   int   # context window size
    embd_dim:   int   # embedding dimension
    n_layer:    int   # number of Transformer blocks/layers
    n_head:     int   # number of attention qury heads
    n_kv_head:  int   # number of attention key/value heads
    head_dim:   int   # dimension of each attention head
    hidden_dim: int   # hidden dimension of the feedforward layer
    norm_eps:   float # epsilon for RMS layer normalization
    vocab_size: int   # size of the vocabulary
    p_drop:     float # dropout probability in the model

    max_batch_size: int = 0

    # For rotary embeddings. If not set, will be 1e4 as the default value.
    rope_theta: Optional[float] = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe: Optional[MoeArgs] = None


# utility functions for Rotary Positional Embedding
def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    q_out = torch.view_as_real(q_ * freqs_cis).flatten(-2)
    k_out = torch.view_as_real(k_ * freqs_cis).flatten(-2)
    return q_out.to(q.dtype), k_out.to(k.dtype)


class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the
    end. It is possible to use torch.nn.MultiheadAttention here but I am 
    including an explicit implementation here to show that there is nothing too 
    scary here.
    """

    def __init__(self, config):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.embd_dim, 3 * config.head_dim)
        # output projection
        self.c_proj = nn.Linear(config.head_dim, config.embd_dim)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in
        # the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.cxt_size, config.cxt_size)
                      ).view(1, 1, config.cxt_size, config.cxt_size)
        )
        self.n_head = config.n_head
        self.embd_dim = config.embd_dim

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (embd_dim)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.head_dim, dim=2)
        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # causal self-attention; Self-attend:
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, self.head_dim)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


## Grouped Attention Layer
class GroupedQueryAttention(nn.Module):
    """
    A grouped-query attention layer with a projection at the end. This class can
    also cover the following cases:
    - Multi-head attention: set `n_kv_head` equal to `n_head`
    - Multi-query attention: set `n_kv_head` equal to 1, and `n_head` to the
        number of query heads
    - Grouped-query attention: set `n_head` to the number of query heads, and
        `n_kv_head` to `n_group = n_head // n_kv_head`.
    """
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.n_head: int = config.n_head
        self.n_kv_head: int = config.n_kv_head
        self.head_dim: int = config.head_dim

        self.repeats = self.n_head // self.n_kv_head
        self.scale = config.head_dim ** -0.5

        self.q_proj = nn.Linear(
            config.embd_dim, config.n_head * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.embd_dim, config.n_kv_head * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.embd_dim, config.n_kv_head * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.n_head * config.head_dim, config.embd_dim, bias=False
        )

        self.p_drop = config.p_drop if config.p_drop is not None else 0.1

        self.register_buffer(
            "attn_bias",
            torch.tril(
                torch.ones(config.cxt_size, config.cxt_size)
            ).view(1, 1, config.cxt_size, config.cxt_size)
        )

    def forward(self, x: torch.Tensor, freq_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        q, k = apply_rotary_emb(q, k, freq_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        k = torch.repeat_interleave(k, repeats=self.repeats, dim=1)
        v = torch.repeat_interleave(v, repeats=self.repeats, dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(
            self.attn_bias[:, :, :T, :T] == 0, float("-inf")
        )
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, self.p_drop, training=self.training)
        y = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, -1)
        y = F.dropout(y, self.p_drop, training=self.training)
        return self.o_proj(y)


class MoeLayer(nn.Module):
    def __init__(
        self,
        experts: List[nn.Module],
        gate: nn.Module,
        config: MoeArgs
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.config = config

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(input)
        weights, selected_experts = torch.topk(
            gate_logits, self.config.num_experts_per_tok
        )
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(input.dtype)
        results = torch.zeros_like(input)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert] * expert(
                input[batch_idx]
            )
        return results


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.h1 = nn.Linear(config.embd_dim, config.hidden_dim, bias=False)
        self.h2 = nn.Linear(config.hidden_dim, config.embd_dim, bias=False)
        self.h3 = nn.Linear(config.embd_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h2(F.silu(self.h1(x)) * self.h3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class GPTBlock(nn.Module):
    """Transformer block used by GPT-2"""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embd_dim)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.embd_dim)
        self.ffn = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.embd_dim, 4 * config.embd_dim),
            c_proj  = nn.Linear(4 * config.embd_dim, config.embd_dim),
            act     = nn.GELU(),
            dropout = nn.Dropout(config.p_drop),
        ))
        m = self.ffn
        # FFN forward
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class MistralBlock(nn.Module):
    """Transformer block used by Mistral"""
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.embd_dim = config.embd_dim
        self.attn = GroupedQueryAttention(config)
        self.attn_norm = RMSNorm(config.embd_dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.embd_dim, eps=config.norm_eps)

        self.ffn: nn.Module
        if config.moe is not None:
            self.ffn = MoeLayer(
                experts=[
                    FeedForward(config) for _ in range(config.moe.num_experts)
                ],
                gate=nn.Linear(
                    config.embd_dim, config.moe.num_experts, bias=False
                ),
                moe_config=config.moe,
            )
        else:
            self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, freq_cis: torch.Tensor) -> torch.Tensor:
       r = self.attn(self.attn_norm(x), freq_cis)
       h = x + r
       r = self.ffn(self.ffn_norm(h))
       return h + r
