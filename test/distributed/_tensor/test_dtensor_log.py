# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import functools
import os
import unittest

import torch.distributed as dist
from torch._dynamo.test_case import run_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.logging_utils import LoggingTestCase

requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


class DTensorLogTest(LoggingTestCase):
    @requires_distributed()
    @skip_if_lt_x_gpu(2)
    def test_dtensor_log(self):
        env = dict(os.environ)
        env["TORCH_LOGS"] = "+dtensor"
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["MASTER_PORT"] = "12345"
        env["MASTER_ADDR"] = "localhost"

        stdout, stderr = self.run_process_no_exception(
            """\
import logging
import torch
from torch.distributed._tensor import  init_device_mesh, distribute_tensor, Shard

mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
placements = [Shard(0)]
tensor = torch.randn(12, 8, 8)
dtensor = distribute_tensor(tensor, mesh, placements)
dtensor.max()
""",
            env=env,
        )
        self.assertIn("_dispatch.py", stderr.decode("utf-8"))
        self.assertIn("redistribute=False", stderr.decode("utf-8"))


if __name__ == "__main__":
    run_tests()
