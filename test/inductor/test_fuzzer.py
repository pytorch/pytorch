# Owner(s): ["module: dynamo"]

import unittest
from typing import List, Literal

import torch
from torch._inductor import config
from torch._inductor.fuzzer import ConfigFuzzer, SamplingMethod
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal import fake_config_module as fake_config
from torch.testing._internal.inductor_utils import HAS_GPU


def create_simple_test_model_cpu():
    def test_fn() -> bool:
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
        )

        x = torch.randn(32, 10)
        y = model(x)
        return True

    return test_fn


def create_simple_test_model_gpu():
    batch_size = 32
    seq_length = 50
    hidden_size = 768

    inp = torch.randn(batch_size, seq_length, hidden_size, device="cuda")
    weight = torch.randn(hidden_size, hidden_size, device="cuda")

    def test_fn() -> bool:
        matmul_output = inp @ weight
        final_output = torch.nn.LayerNorm(hidden_size, device="cuda")(matmul_output)
        return True

    return test_fn


class TestConfigFuzzer(TestCase):
    def test_sampling_method_toggle(self):
        toggle = SamplingMethod.dispatch(SamplingMethod.TOGGLE)
        self.assertEqual(toggle(bool, False), True)
        self.assertEqual(toggle(bool, True), False)
        self.assertEqual(toggle(Literal["foo", "bar"], "foo"), "bar")
        self.assertEqual(toggle(Literal["foo", "bar"], "bar"), "foo")
        self.assertTrue("bar" in toggle(List[Literal["foo", "bar"]], ["foo"]))
        self.assertTrue("foo" in toggle(List[Literal["foo", "bar"]], ["bar"]))

    def test_sampling_method_random(self):
        random = SamplingMethod.dispatch(SamplingMethod.RANDOM)
        samp = [random(bool, False) for i in range(1000)]
        self.assertTrue(not all(samp))

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_config_fuzzer_inductor_gpu(self):
        fuzzer = ConfigFuzzer(config, create_simple_test_model_gpu, seed=30)
        self.assertIsNotNone(fuzzer.default)
        fuzzer.reproduce([{"max_fusion_size": 1}])

    def test_config_fuzzer_inductor_cpu(self):
        fuzzer = ConfigFuzzer(config, create_simple_test_model_cpu, seed=100)
        self.assertIsNotNone(fuzzer.default)
        fuzzer.reproduce([{"max_fusion_size": 1}])

    def test_config_fuzzer_bisector(self):
        key_1 = {"e_bool": False, "e_optional": None}

        class MyException(Exception):
            pass

        def create_key_1():
            def myfn():
                if not fake_config.e_bool and fake_config.e_optional is None:
                    raise MyException("hi")
                return True

            return myfn

        fuzzer = ConfigFuzzer(fake_config, create_key_1, seed=100, default={})
        results = fuzzer.bisect(num_attempts=2, p=1.0)
        self.assertEqual(len(results), 2)
        for res in results:
            self.assertEqual(res, key_1)


if __name__ == "__main__":
    run_tests()
