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
import math

from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F

from modules import GPTBlock, MistralBlock, ModelArgs, RMSNorm
from modules import precompute_freqs_cis


class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = ModelArgs()
        # Config must give either model_type or
        # (embd_dim, n_layer, n_head)
        C.model_type = 'gpt'
        C.cxt_size = None
        C.embd_sim = None
        C.n_layer = None
        C.n_head = None
        C.n_kv_head = None
        C.head_dim = None
        C.hidden_dim = None
        C.norm_eps = None
        C.vocab_size = None
        C.p_drop = 0.1
        C.max_batch_size = None
        C.rope_theta = None
        C.moe = None
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.cxt_size is not None
        self.cxt_size = config.cxt_size

        type_given = config.model_type is not None
        params_given = all([
            config.embd_dim is not None,
            config.n_layer is not None,
            config.n_head is not None
        ])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-2 configs
                # 117M params
                'openai-gpt':   dict(n_layer=12, n_head=12, embd_dim=768),
                # 124M params
                'gpt2':         dict(n_layer=12, n_head=12, embd_dim=768),
                # 350M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, embd_dim=1024),
                # 774M params
                'gpt2-large':   dict(n_layer=36, n_head=20, embd_dim=1280),
                # 1558M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, embd_dim=1600),
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, embd_dim=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, embd_dim=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, embd_dim=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, embd_dim=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embd_dim),
            wpe = nn.Embedding(config.cxt_size, config.embd_dim),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList(
                [GPTBlock(config) for _ in range(config.n_layer)]
            ),
            ln_f = nn.LayerNorm(config.embd_dim),
        ))
        self.lm_head = nn.Linear(config.embd_dim, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual
        # projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean = 0.0, std = 0.02 / math.sqrt(2 * config.n_layer)
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

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.cxt_size, \
            f'Cannot forward sequence of length {t},' + \
                f'block size is only {self.cxt_size}'
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) 

        # forward the GPT model itself
        # token embeddings of shape (b, t, embd_dim)
        tok_emb = self.transformer.wte(idx)
        # position embeddings of shape (1, t, embd_dim)
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
    def generate(
        self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None
    ) -> torch.LongTensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and complete the sequence max_new_tokens times, feeding the predictions
        back into the model each time. Most likely you'll want to make sure to
        be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at
            # cxt_size
            idx_cond = idx if idx.size(1) <= self.cxt_size else \
                              idx[:, -self.cxt_size:]
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

class Mistral(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        assert config.vocab_size is not None
        assert config.cxt_size is not None
        self.cxt_size = config.cxt_size
        self.vocab_size = config.vocab_size

        type_given = config.model_type is not None
        params_given = all([
            config.embd_dim is not None,
            config.n_layer is not None,
            config.n_head is not None,
            config.n_kv_head is not None,
            config.head_dim is not None,
            config.hidden_dim is not None,
        ])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict(
                {# names follow the huggingface naming conventions
                 # Mistral configs
                 # 7B params
                    'mistral': dict(
                         cxt_size=1024,
                         n_layer=32,
                         n_head=32,
                         n_kv_head=8,
                         embd_dim=768,
                         head_dim=14336,
                         hidden_dim=4096,
                         vocab_size=32000,
                         p_drop = 0.1,
                         norm_eps=1e-5,
                         rope_theta=1e4,
                         moe=None,
                    ),
                     # 0.47m params
                     'mistral-nano': dict(
                         cxt_size=32,
                         n_layer=4,
                         n_head=8,
                         n_kv_head=2,
                         embd_dim=32,
                         head_dim=128,
                         hidden_dim=128,
                         vocab_size=3000,
                         p_drop = 0.1,
                         norm_eps=1e-5,
                         rope_theta=1e4,
                         moe=None,
                     ),
                }[config.model_type]
            )

        self.n_layer = config.n_layer
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None
        assert self.vocab_size > 0

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embd_dim),
            drop = nn.Dropout(config.p_drop),
            h = nn.ModuleList(
                [MistralBlock(config) for _ in range(config.n_layer)]
            ),
            ln_f = RMSNorm(config.embd_dim, eps=config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.embd_dim, config.vocab_size, bias=False)

        # (note we don't count the parameters in lm_head)
        self.n_params = sum(p.numel() for p in self.transformer.parameters())

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freq_cis(self) -> torch.Tensor:
        if self._precomputed_freqs_cis is None:
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.config.embd_dim,
                self.config.cxt_size,
                self.config.rope_theta
            )
        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                self.device
            )
        return self._precomputed_freqs_cis

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()

        assert t <= self.cxt_size, \
            f'Cannot forward sequence of length {t},' + \
                f'block size is only {self.cxt_size}'
        assert b == 1, \
            f'Only batch size 1 is supported, got {b}'

        _freq_cis = self._precomputed_freqs_cis[:t, :]

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x, _freq_cis)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits
