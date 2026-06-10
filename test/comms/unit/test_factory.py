#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

import torch
import torch.comms
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFactory(TestCase):
    def test_factory(self):
        print(torch.comms)
        print(dir(torch.comms))

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["TORCHCOMM_GLOO_HOSTNAME"] = "localhost"

        comm = torch.comms.new_comm("gloo", torch.device("cpu"), "my_comm")
        comm.finalize()
        backend = comm.get_backend_impl()
        print(backend)

        from torch.comms._comms_gloo import TorchCommGloo

        # if backend was lazily loaded backend will not have the right type
        self.assertIsInstance(backend, TorchCommGloo)

    def test_factory_missing(self):
        with self.assertRaisesRegex(ModuleNotFoundError, "failed to find backend"):
            torch.comms.new_comm("invalid", torch.device("cuda"), "my_comm")


if __name__ == "__main__":
    run_tests()
