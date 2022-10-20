#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from copy import deepcopy

import torch
import torch.nn as nn

from torch.distributed.optim import apply_optimizer_in_backward

# TODO (rohan-varma): Add FSDP & DDP tests once supported


def _validate_params(params_1, params_2, fn):
    for p1, p2 in zip(params_1, params_2):
        fn(p1, p2)


class ApplyOverlappedOptimizerTest(unittest.TestCase):
    def test_apply_optimizer_in_backward(self) -> None:
        optimizer_kwargs = {"lr": 1.0}
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        optim = torch.optim.SGD(model.parameters(), **optimizer_kwargs)
        model_with_opt_in_bwd = deepcopy(model)

        apply_optimizer_in_backward(
            torch.optim.SGD,
            model_with_opt_in_bwd.parameters(),
            optimizer_kwargs=optimizer_kwargs,
        )

        _validate_params(
            model.parameters(),
            model_with_opt_in_bwd.parameters(),
            torch.testing.assert_allclose,
        )

        for i in range(6):
            inp = torch.randn(4, 10)
            model(inp).sum().backward()
            model_with_opt_in_bwd(inp.clone()).sum().backward()
            optim.step()
            with self.subTest(i):
                _validate_params(
                    model.parameters(),
                    model_with_opt_in_bwd.parameters(),
                    torch.testing.assert_allclose,
                )
            # For equivalence since run in backwards sets it to none.
            optim.zero_grad(set_to_none=True)
