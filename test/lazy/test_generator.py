# Owner(s): ["oncall: jit"]

import torch
import torch._lazy.metrics as metrics
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


torch._lazy.ts_backend.init()


class LazyGeneratorTest(TestCase):
    def test_generator(self):
        """
        Test that generators are being inserted into the TorchScript
        graph by setting different seeds before each call to
        generate_tensor but the resulting tensor is the same
        """

        def generate_tensor():
            g1 = torch.Generator()
            g1.manual_seed(2023)
            t1 = torch.tensor(1.0)
            t1.uniform_(generator=g1)

            g2 = torch.Generator()
            g2.manual_seed(2024)
            t2 = torch.tensor(1.0)
            t2.normal_(generator=g2)

            return t1, t2

        torch.manual_seed(1)

        with torch.device("cpu"):
            cpu_t1, cpu_t2 = generate_tensor()

        torch.manual_seed(2)

        with torch.device("lazy"):
            lazy_t1, lazy_t2 = generate_tensor()

        torch._lazy.mark_step()

        assert torch.allclose(
            cpu_t1, lazy_t1.to("cpu")
        ), f"Expected {cpu_t1}, got {lazy_t1.to('cpu')}"
        assert torch.allclose(
            cpu_t2, lazy_t2.to("cpu")
        ), f"Expected {cpu_t2}, got {lazy_t2.to('cpu')}"

    @skipIfTorchDynamo("Torch Dynamo does not support torch.Generator type")
    def test_generator_causes_multiple_compiles(self):
        """
        Test that inserting generators with different seed caused recompile
        """

        def generate_tensor(seed):
            t = torch.tensor(1.0)
            g = torch.Generator()
            g.manual_seed(seed)
            t.uniform_(-1, 1, generator=g)
            return t

        metrics.reset()

        with torch.device("lazy"):
            t = generate_tensor(1)
            torch._lazy.mark_step()

            uncached_compile = metrics.counter_value("UncachedCompile")
            assert (
                uncached_compile == 1
            ), f"Expected 1 uncached compiles, got {uncached_compile}"

            t = generate_tensor(2)
            torch._lazy.mark_step()

            uncached_compile = metrics.counter_value("UncachedCompile")
            assert (
                uncached_compile == 2
            ), f"Expected 2 uncached compiles, got {uncached_compile}"

            t = generate_tensor(1)
            torch._lazy.mark_step()

            uncached_compile = metrics.counter_value("UncachedCompile")
            assert (
                uncached_compile == 2
            ), f"Expected 2 uncached compiles, got {uncached_compile}"
            cached_compile = metrics.counter_value("CachedCompile")
            assert (
                cached_compile == 1
            ), f"Expected 1 cached compile, got {cached_compile}"

        metrics.reset()

        latest_graph = torch._C._lazy_ts_backend._get_latest_computation_graph()
        assert 'torch.Generator(device="cpu", seed=1)' in latest_graph
        assert "aten::uniform" in latest_graph


if __name__ == "__main__":
    run_tests()
