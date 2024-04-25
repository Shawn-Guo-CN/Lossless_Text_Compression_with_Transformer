"""
Definition of a GPT language model and a Mistral language model.

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

from typing import Optional, Union
import torch
import torch.nn as nn
from simple_parsing.helpers import Serializable
from torch.nn import functional as F

from modules import GPTBlock


@dataclass
class ModelArgs(Serializable):
    model_type: Union[str, None]
    n_layer: int
    n_head: int
    n_embd: int
    norm_eps: float
    vocab_size: int
    block_size: int
    embd_pdrop: float
    resid_pdrop: float
    attn_pdrop: float





@dataclass
class ModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int

    max_batch_size: int = 0

    # For rotary embeddings. If not set, will be infered from sliding window.
    rope_theta: Optional[float] = None
    # If this is set, use sliding window attention rotating cache.
    sliding_window: Optional[int] = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe: Optional[MoeArgs] = None

# -----------------------------------------------------------------------------

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = ModelArgs()
        # either model_type or
        # (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([
            config.n_layer is not None,
            config.n_head is not None,
            config.n_embd is not None
        ])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions GPT-1
                # GPT-2 configs
                # 117M params
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),
                # 124M params
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
                # 350M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
                # 774M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
                # 1558M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual
        # projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        # (note we don't count the decoder parameters in lm_head)
        self.n_params = sum(p.numel() for p in self.transformer.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, \
            f'Cannot forward sequence of length {t},' + \
                f'block size is only {self.block_size}'
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) 

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)
        # position embeddings of shape (1, t, n_embd)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def get_loss(self, logits, targets):
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at
            # block_size
            idx_cond = idx if idx.size(1) <= self.block_size else \
                              idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired
            # temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely
            # element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# -----------------------------------------------------------------------------


