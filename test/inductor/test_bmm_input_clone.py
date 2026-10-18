# Owner(s): ["module: inductor"]
import torch
from torch._inductor import config, metrics
from torch._inductor.fx_passes.post_grad import remove_noop_ops
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification, parametrize


class TestBmmInputClone(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def _inputs(self, device, broadcast=False):
        # Integer-valued half inputs make the dot products exactly representable
        # with both contiguous and padded layouts.
        batches = 1 if broadcast else 3
        storage = torch.arange(
            batches * 13 * 7, dtype=torch.float32, device=device
        ).reshape(batches, 13, 7)
        padded = (storage.remainder(3) - 1).to(torch.float16)[:, :9, :]
        if broadcast:
            padded = padded.expand(3, -1, -1)
        weight = torch.arange(3 * 7 * 5, device=device).reshape(3, 7, 5)
        return padded, (weight.remainder(3) - 1).to(torch.float16)

    def _check(self, fn, inputs, expected_clones, enabled=True):
        graph_module = make_fx(fn)(*inputs)
        expected = graph_module(*(x.clone() for x in inputs))
        with config.patch(bmm_elide_padded_input_clones=enabled):
            remove_noop_ops(graph_module.graph)
        graph_module.graph.lint()
        graph_module.recompile()
        clones = sum(
            node.target is torch.ops.aten.clone.default
            for node in graph_module.graph.nodes
        )
        self.assertEqual(clones, expected_clones)
        # Use the original padded layout when executing the rewritten graph.
        self.assertEqual(graph_module(*inputs), expected, atol=0, rtol=0)

    @parametrize("broadcast", (False, True))
    def test_padded_readonly_bmm(self, device, broadcast):
        def fn(x, weight):
            return torch.bmm(x.clone(), weight)

        inputs = self._inputs(device, broadcast)
        self._check(fn, inputs, expected_clones=0 if device.startswith("cuda") else 1)

    def test_disabled(self, device):
        self._check(
            lambda x, weight: torch.bmm(x.clone(), weight),
            self._inputs(device),
            expected_clones=1,
            enabled=False,
        )

    def test_clone_is_also_output(self, device):
        def fn(x, weight):
            copied = x.clone()
            return torch.bmm(copied, weight), copied

        self._check(fn, self._inputs(device), expected_clones=1)

    def test_non_bmm_consumer(self, device):
        def fn(x, weight):
            copied = x.clone()
            return torch.bmm(copied, weight), copied.sum()

        self._check(fn, self._inputs(device), expected_clones=1)

    def test_source_mutation_preserves_snapshot(self, device):
        def fn(x, weight):
            copied = x.clone()
            x.copy_(x + 1)
            return torch.bmm(copied, weight)

        self._check(fn, self._inputs(device), expected_clones=1)

    def test_noncontiguous_columns(self, device):
        x, weight = self._inputs(device)
        storage = torch.empty(3, 9, 14, device=device, dtype=torch.float16)
        x_with_gaps = storage[:, :, ::2]
        x_with_gaps.copy_(x)
        self._check(
            lambda x, weight: torch.bmm(x.clone(), weight),
            (x_with_gaps, weight),
            expected_clones=1,
        )


instantiate_device_type_tests(TestBmmInputClone, globals(), except_for=("mps", "xpu"))


class TestBmmInputCloneCompiled(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_compiled_copy_elision(self, device):
        def fn(x, weight):
            return torch.bmm(x.clone(), weight)

        storage = torch.arange(3 * 13 * 7, device=device).reshape(3, 13, 7)
        x = (storage.remainder(3) - 1).to(torch.float16)[:, :9, :]
        weight = torch.ones(3, 7, 5, device=device, dtype=torch.float16)
        expected = fn(x, weight)
        counts = {}
        for enabled in (False, True):
            torch._dynamo.reset()
            metrics.reset()
            with config.patch(
                bmm_elide_padded_input_clones=enabled,
                force_disable_caches=True,
                max_autotune=False,
                max_autotune_gemm=False,
            ):
                actual, _ = run_and_get_code(
                    torch.compile(fn, fullgraph=True), x, weight
                )
                self.assertEqual(actual, expected, atol=0, rtol=0)
                counts[enabled] = metrics.generated_kernel_count
        self.assertGreater(counts[False], 0)
        self.assertEqual(counts[True], 0)


instantiate_device_type_tests(
    TestBmmInputCloneCompiled, globals(), except_for=("cpu", "mps", "xpu")
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
