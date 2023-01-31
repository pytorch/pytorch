# Copyright (c) Meta Platforms, Inc. and affiliates

import torch

import torch.distributed._tensor.api as dtensor

def unwrap_local_tensor(e: "dtensor.DTensor") -> torch.Tensor:
    return e._local_tensor if isinstance(e, dtensor.DTensor) else e
