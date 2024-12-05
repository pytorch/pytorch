# Owner(s): ["module: fsdp"]
import functools
import os
import unittest.mock

import torch.distributed as dist
from torch._dynamo.test_case import run_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.testing._internal.logging_utils import LoggingTestCase


requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")
requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


@skip_if_lt_x_gpu(2)
class LoggingTests(LoggingTestCase):
    @requires_distributed()
    def test_fsdp_logging(self):
        env = dict(os.environ)
        env["TORCH_LOGS"] = "fsdp"
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["MASTER_PORT"] = "34715"
        env["MASTER_ADDR"] = "localhost"
        stdout, stderr = self.run_process_no_exception(
            """\
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
logger = logging.getLogger("torch.distributed._composable.fsdp")
logger.setLevel(logging.DEBUG)
device = "cuda"
torch.manual_seed(0)
model = nn.Sequential(*[nn.Linear(4, 4, device=device, bias=False) for _ in range(2)])
for layer in model:
    fully_shard(layer)
fully_shard(model)
x = torch.randn((4, 4), device=device)
model(x).sum().backward()
""",
            env=env,
        )
        self.assertIn("FSDP::root_pre_forward", stderr.decode("utf-8"))
        self.assertIn("FSDP::pre_forward (0)", stderr.decode("utf-8"))
        self.assertIn("FSDP::pre_forward (1)", stderr.decode("utf-8"))
        self.assertIn("FSDP::post_forward (0)", stderr.decode("utf-8"))
        self.assertIn("FSDP::post_forward (1)", stderr.decode("utf-8"))
        self.assertIn("FSDP::pre_backward (0)", stderr.decode("utf-8"))
        self.assertIn("FSDP::pre_backward (1)", stderr.decode("utf-8"))
        self.assertIn("FSDP::post_backward (0)", stderr.decode("utf-8"))
        self.assertIn("FSDP::post_backward (1)", stderr.decode("utf-8"))
        self.assertIn("FSDP::root_post_backward", stderr.decode("utf-8"))


if __name__ == "__main__":
    run_tests()
