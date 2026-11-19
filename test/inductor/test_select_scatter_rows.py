# Owner(s): ["module: inductor"]

import torch
from torch._dynamo.utils import counters
from torch._guards import detect_fake_mode
from torch._inductor import config
from torch._inductor.fx_passes.post_grad import pass_patterns
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize


aten = torch.ops.aten


class TestSelectScatterRows(TestCase):
    @config.patch(force_disable_caches=True)
    @parametrize("indices", [(2, 0, 3), (-1, 1, 0), (3, 1, 0, 2)])
    @parametrize("computed", [False, True])
    def test_assemble_rows(self, device, indices, computed):
        def fn(base, *updates):
            for index, update in zip(indices, updates):
                if computed:
                    update = update.relu()
                base = base.select_scatter(update, -2, index)
            return base

        # The four-row case replaces every row. The other cases retain gaps.
        height = 4 if len(indices) == 4 else 128
        base = torch.randn(height, 65_536 // height, device=device)
        original = base.clone()
        updates = [torch.randn_like(base[0]) for _ in indices]
        expected = fn(base, *updates)
        before = counters["inductor"]["select_scatter_row_chains"]
        actual = torch.compile(fn, fullgraph=True)(base, *updates)
        self.assertEqual(counters["inductor"]["select_scatter_row_chains"], before + 1)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.stride(), expected.stride())
        actual.zero_()
        self.assertEqual(base, original)

    def check_unrewritten(self, fn, args, tracing_mode="fake"):
        gm = make_fx(fn, tracing_mode=tracing_mode)(*args)

        def scatters():
            return list(
                gm.graph.find_nodes(
                    op="call_function", target=aten.select_scatter.default
                )
            )

        original = scatters()
        self.assertTrue(original)
        fake_mode = detect_fake_mode([n.meta.get("val") for n in gm.graph.nodes])
        with V.set_fake_mode(fake_mode):
            pass_patterns[1].apply(gm)
        gm.graph.lint()
        self.assertEqual(scatters(), original)

    @parametrize("indices", [(1, 2), (1, 0, -255), (1, 4, 7), tuple(range(9))])
    def test_skip_short_overlapping_fragmented_or_long_chain(self, device, indices):
        def fn(base, row):
            for i in indices:
                base = base.select_scatter(row, 0, i)
            return base

        args = (torch.randn(256, 256, device=device), torch.randn(256, device=device))
        self.check_unrewritten(fn, args)

    @parametrize(
        "case",
        [
            "layout",
            "source_stride",
            "cast",
            "column",
            "small",
            "symbolic",
            "prefix_storage",
            "offset",
        ],
    )
    def test_skip_unsupported_inputs(self, device, case):
        dim = 1 if case == "column" else 0

        def fn(base, row):
            for i in (2, 0, 3):
                base = base.select_scatter(row, dim, i)
            return base

        size = 8 if case == "small" else 256
        base = torch.randn(size, size, device=device)
        row = torch.randn(size, device=device)
        if case == "layout":
            base = base.t()
        elif case in ("prefix_storage", "offset"):
            storage = torch.randn(size + 1, size, device=device)
            base = storage[:size] if case == "prefix_storage" else storage[1:]
        elif case == "source_stride":
            row = base[:, 0]
        elif case == "cast":
            row = row.double()
        mode = "symbolic" if case == "symbolic" else "fake"
        self.check_unrewritten(fn, (base, row), tracing_mode=mode)

    @parametrize(
        "case", ["live", "dependent", "mutation", "shared_producer", "consumer"]
    )
    def test_preserve_dataflow(self, device, case):
        def fn(base, a, b):
            if case == "shared_producer":
                a = a.sin()
            first = base.select_scatter(a, 0, 2)
            if case == "mutation":
                # Delaying the first read until cat would observe the new value.
                a.add_(1)
            if case == "dependent":
                b = first[0]
            middle = first.select_scatter(b, 0, 0)
            last = middle.select_scatter(a, 0, 3)
            if case == "live":
                return first, last
            return last.cos() if case == "consumer" else last

        base = torch.randn(256, 256, device=device)
        args = (base, torch.randn_like(base[0]), torch.randn_like(base[0]))
        self.check_unrewritten(fn, args)


instantiate_device_type_tests(TestSelectScatterRows, globals(), only_for=("cpu",))

if __name__ == "__main__":
    run_tests()
