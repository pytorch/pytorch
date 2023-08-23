# Owner(s): ["module: inductor"]
import random
import time

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.test_case import TestCase

from torch.testing._internal.common_utils import set_rng_seed, TEST_WITH_ASAN
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

aten = torch.ops.aten


class Foo(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.linear = nn.Linear(15, 30)

    def forward(self, x: torch.Tensor):
        y = x * x
        y = self.linear(y)
        y += 3
        y -= 1
        return y


used_cache_keys = []


def activate_unique_cache():
    new_cache_key = random.randint(0, 1000000000)
    while new_cache_key in used_cache_keys:
        new_cache_key = random.randint(0, 1000000000)
    used_cache_keys.append(new_cache_key)
    inductor_config.fx_graph_cache = new_cache_key


class CacheTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._saved_config = inductor_config.save_config()

    def tearDown(self):
        super().tearDown()
        inductor_config.load_config(self._saved_config)

    def setUp(self):
        super().setUp()
        # We don't want a fixed seed for this test to avoid cache collisions
        set_rng_seed(int(time.time()))


if HAS_CPU:

    class TestCodeGenCache(CacheTestCase):
        def test_cache_function_output(self):
            activate_unique_cache()

            @torch.compile
            def foo(input, weight, bias):
                return F.relu(F.linear(input, weight, bias))

            input = torch.randn(64, 32)
            weight = torch.randn(16, 32)
            bias = torch.randn(16)

            base_result = foo(input, weight, bias)
            base_compilation_metrics = torch._dynamo.utils.compilation_time_metrics

            # The compilation metrics of the first compilation should show codegen
            self.assertIn("Scheduler.codegen", base_compilation_metrics)

            # Important to reset so in-memory guards don't prevent compilation
            torch._dynamo.reset()

            cached_result = foo(input, weight, bias)
            cached_compilation_metrics = torch._dynamo.utils.compilation_time_metrics

            # The cached_result coming from a cached kernel should be the same as the original one
            # The compilation metrics of the cached compilation should show no codegen
            self.assertEqual(base_result, cached_result)
            self.assertNotIn("Scheduler.codegen", cached_compilation_metrics)

        def test_cache_module_output(self):
            activate_unique_cache()

            model = torch.compile(Foo())

            input = torch.rand(15, 15)

            base_result = model(input)
            base_compilation_metrics = torch._dynamo.utils.compilation_time_metrics

            # The compilation metrics of the first compilation should show codegen
            self.assertIn("Scheduler.codegen", base_compilation_metrics)

            # Important to reset so in-memory guards don't prevent compilation
            torch._dynamo.reset()

            cached_result = model(input)
            cached_compilation_metrics = torch._dynamo.utils.compilation_time_metrics
            # The cached_result coming from a cached kernel should be the same as the original one
            # The compilation metrics of the cached compilation should show no codegen
            self.assertEqual(base_result, cached_result)
            self.assertNotIn("Scheduler.codegen", cached_compilation_metrics)


if HAS_CUDA and not TEST_WITH_ASAN:

    class TestCodeGenCacheCuda(CacheTestCase):
        def test_cache_cudagraphs(self):
            activate_unique_cache()

            model = Foo()

            cuda_model1 = model.cuda()
            opt_model1 = torch.compile(cuda_model1, mode="reduce-overhead")

            input = torch.rand(15, 15).cuda()

            base_result = opt_model1(input).clone().detach()
            base_compilation_metrics = torch._dynamo.utils.compilation_time_metrics

            # The compilation metrics of the first compilation should show codegen
            self.assertIn("Scheduler.codegen", base_compilation_metrics)

            # Important to reset so in-memory guards don't prevent compilation
            torch._dynamo.reset()

            input2 = torch.rand(15, 15).cuda()
            cuda_model2 = Foo().cuda()
            opt_model2 = torch.compile(cuda_model2, mode="reduce-overhead")
            cached_result = opt_model2(input2)
            cached_compilation_metrics = torch._dynamo.utils.compilation_time_metrics
            # The cached_result coming from a cached kernel should be the same as the original one
            # The compilation metrics of the cached compilation should show no codegen
            self.assertEqual(base_result, cached_result)
            self.assertNotIn("Scheduler.codegen", cached_compilation_metrics)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
