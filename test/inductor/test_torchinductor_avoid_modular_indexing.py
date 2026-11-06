# Owner(s): ["module: inductor"]
import importlib
import os
import sys

from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    CommonTemplate,
    copy_tests,
)


importlib.import_module("filelock")

AvoidModularIndexingCommonTemplate = make_test_cls_with_patches(
    CommonTemplate,
    "AvoidModularIndexing",
    "_avoid_modular_indexing",
    (config, "avoid_modular_indexing", True),
)


if HAS_CPU:

    class AvoidModularIndexingCpuTests(TestCase):
        common = check_model
        device = "cpu"

    copy_tests(AvoidModularIndexingCommonTemplate, AvoidModularIndexingCpuTests, "cpu")

if HAS_GPU and not TEST_WITH_ASAN:

    class AvoidModularIndexingGPUTests(TestCase):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(
        AvoidModularIndexingCommonTemplate, AvoidModularIndexingGPUTests, GPU_TYPE
    )


if __name__ == "__main__":
    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
