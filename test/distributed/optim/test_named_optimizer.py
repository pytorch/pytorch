# Owner(s): ["oncall: distributed"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torch.distributed.optim import _NamedOptimizer


class TestDummyModel(torch.nn.Module):
    def __init__(self):
        super(TestDummyModel, self).__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))


class NamedOptimizerTest(unittest.TestCase):
    def _compare_state_dict_group(self, group, named_group, assert_equal=True):
        for key, val in group.items():
            if key != "params":
                self.assertTrue(
                    key in named_group, f"{key} not in named optimizer state dict"
                )
                err_msg = (
                    f"{key} state not equal" if assert_equal else f"{key} state equal"
                )
                if isinstance(val, torch.Tensor):
                    fn = self.assertTrue if assert_equal else self.assertFalse
                    fn(torch.allclose(val, named_group[key]), err_msg)
                else:
                    fn = self.assertEqual if assert_equal else self.assertNotEqual
                    fn(val, named_group[key], err_msg)

    def test_state_dict(self):
        """Check that NamedOptimizer exposes the expected state dict
        interface."""
        m = TestDummyModel()
        m_dup = TestDummyModel()
        optim_1 = torch.optim.SGD(
            [
                {"params": m.net1.parameters()},
                {"params": m.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )

        optim_2 = torch.optim.Adam(
            [
                {"params": m.net2.parameters()},
                {"params": m.net4.parameters(), "lr": 1e-5},
            ]
        )

        named_optim_1 = _NamedOptimizer(
            m_dup.named_parameters(),
            torch.optim.SGD,
            [
                {"params": m_dup.net1.parameters()},
                {"params": m_dup.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )

        named_optim_2 = _NamedOptimizer(
            m_dup.named_parameters(),
            torch.optim.Adam,
            [
                {"params": m_dup.net2.parameters()},
                {"params": m_dup.net4.parameters(), "lr": 1e-5},
            ],
        )
        for i in range(2):
            x = torch.rand(5, 8)
            y = m(x)
            y.sum().backward()
            optim_1.step()
            optim_2.step()

            y = m_dup(x)
            y.sum().backward()
            named_optim_1.step()
            named_optim_2.step()

        sd_1 = optim_1.state_dict()
        sd_2 = optim_2.state_dict()
        named_sd_1 = named_optim_1.state_dict()
        named_sd_2 = named_optim_2.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            sd_1["state"][0],
            named_sd_1["state"]["net1.0.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_2["state"][1],
            named_sd_2["state"]["net2.0.bias"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_1["state"][2],
            named_sd_1["state"]["net3.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_2["state"][3],
            named_sd_2["state"]["net4.1.bias"],
            assert_equal=True,
        )

        # Compare "param_groups" in optim state dict
        self._compare_state_dict_group(
            sd_1["param_groups"][0],
            named_sd_1["param_groups"][0],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_2["param_groups"][1], named_sd_2["param_groups"][1], assert_equal=True
        )

    def test_load_state_dict(self):
        """Check that NamedOptimizer exposes the expected state dict
        interface."""
        m = TestDummyModel()
        named_optim_1 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.9,
        )

        for _ in range(2):
            x = torch.rand(5, 8)
            y = m(x)
            y.sum().backward()
            named_optim_1.step()

        state_dict_to_load = named_optim_1.state_dict()

        named_optim_2 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.6,
        )

        for _ in range(2):
            x = torch.rand(5, 8)
            y = m(x)
            y.sum().backward()
            named_optim_2.step()

        state_dict_before_load = named_optim_2.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net1.0.weight"],
            state_dict_before_load["state"]["net1.0.weight"],
            assert_equal=False,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net2.0.bias"],
            state_dict_before_load["state"]["net2.0.bias"],
            assert_equal=False,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net3.weight"],
            state_dict_before_load["state"]["net3.weight"],
            assert_equal=False,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net4.1.bias"],
            state_dict_before_load["state"]["net4.1.bias"],
            assert_equal=False,
        )

        named_optim_2.load_state_dict(state_dict_to_load)
        state_dict_after_load = named_optim_2.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net1.0.weight"],
            state_dict_after_load["state"]["net1.0.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net2.0.bias"],
            state_dict_after_load["state"]["net2.0.bias"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net3.weight"],
            state_dict_after_load["state"]["net3.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net4.1.bias"],
            state_dict_after_load["state"]["net4.1.bias"],
            assert_equal=True,
        )

    def test_load_state_dict_error(self):
        m = TestDummyModel()
        named_optim_1 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.9,
        )

        for _ in range(2):
            x = torch.rand(5, 8)
            y = m(x)
            y.sum().backward()
            named_optim_1.step()

        state_dict_to_load = named_optim_1.state_dict()

        named_optim_2 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.6,
        )

        err_msg = (
            "Expects the optim to be initialized before load but found not initialized"
        )
        with self.assertRaisesRegex(ValueError, err_msg):
            named_optim_2.load_state_dict(state_dict_to_load)
