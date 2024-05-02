"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Optional, Tuple

from simple_parsing.helpers import Serializable
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader


@dataclass
class TrainArgs(Serializable):
    device         : str                 = 'auto'
    num_workers    : int                 = 1
    max_iters      : Optional[int]       = None
    batch_size     : int                 = 1
    learning_rate  : float               = 3e-4
    betas          : Tuple[float, float] = (0.9, 0.95)
    weight_decay   : float               = 0.1
    grad_norm_clip : float               = 1.0


class Trainer(object):
    @staticmethod
    def get_default_config():
        C = TrainArgs()
        return C

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.optimizer = self.configure_optimizers(self.config)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas
        )
        return optimizer

    def update_step(self, batch):
        # setup the optimizer
        self.model.train()

        logits = self.predict_step(batch['x'])
        loss = self.loss_step(logits, batch['y'])
        logits = logits.detach()
        self.optim_step(loss)

        return F.softmax(logits[0][-1], dim=-1)

    def predict_step(self, x):
        x = torch.LongTensor([x]).to(self.device)
        logits = self.model(x)
        return logits

    def loss_step(self, logits, y):
        y = torch.LongTensor([y]).to(self.device)
        return self.model.get_loss(logits, y)

    def optim_step(self, loss):
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_norm_clip
        )
        self.optimizer.step()
