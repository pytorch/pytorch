# Owner(s): ["module: mps"]
import importlib
import os
import sys

import torch


importlib.import_module("filelock")

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model_gpu,
    TestCase,
)


# TODO: Remove this file.
# This tests basic MPS compile functionality


class MPSBasicTests(TestCase):
    common = check_model_gpu
    device = "mps"

    def test_add(self):
        self.common(lambda a, b: a + b, (torch.rand(1024), torch.rand(1024)))

    def test_acos(self):
        self.common(lambda a: a.acos(), (torch.rand(1024),))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if torch.backends.mps.is_available():
        run_tests(needs="filelock")
