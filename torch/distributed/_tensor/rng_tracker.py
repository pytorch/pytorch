# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict

import torch
import torch.distributed as dist

from torch import Tensor


class CudaRNGStatesTracker:
    # we assume that the caller of all the methods would already
    # check if the current device is a CUDA device 
    def __init__(self):
        self._states = {}

    @property
    def rng_states(self) -> Dict[str, Tensor]:
        return self._states

    def rng_state_is_sync(self, name) -> bool:
        return name in self.rng_states

    def sync_rng_state(self) -> None:
        # this function synchronizes CUDA RNG state within GroupMember.WORLD
        if not self.rng_state_is_sync("parallel-rng"):
            cuda_rng_state = torch.cuda.get_rng_state()
            dist.broadcast(cuda_rng_state, 0)
            self.set_rng_state("parallel-rng", cuda_rng_state)

    def set_rng_state(self, name: str, rng_state: Tensor) -> None:
        self.rng_states[name] = rng_state

    def reset(self):
        self.rng_states = {}


_rng_tracker = CudaRNGStatesTracker()
