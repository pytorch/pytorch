# Owner(s): ["module: inductor"]

from unittest import mock

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.fx_passes import singleton_reduction
from torch._inductor.fx_passes.singleton_reduction import (
    eliminate_singleton_reductions,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, TEST_CUDA


LIVE_VOCAB = 4096


class SingletonReductionTests(TestCase):
    def _run_and_check(self, fn, args, expected_fold, patches=None):
        patches = patches or {}
        with config.patch(
            singleton_reduction_elimination=False,
            pattern_matcher=False,
        ):
            expected = torch.compile(fn, fullgraph=True)(*args)

        torch._dynamo.reset()
        counters.clear()
        with config.patch(
            force_disable_caches=True,
            pattern_matcher=False,
            **patches,
        ):
            actual, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True), *args
            )

        self.assertEqual(actual, expected, equal_nan=True)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"],
            int(expected_fold),
        )
        checker = FileCheck()
        if expected_fold:
            checker.check_not("rnumel")
        else:
            checker.check("rnumel")
        checker.run("\n".join(codes))
        return actual, expected

    def _make_pass_graph(self, device):
        def fn(target, scale):
            iota = torch.ops.prims.iota.default(
                LIVE_VOCAB,
                start=0,
                step=1,
                dtype=torch.int64,
                device=torch.device(device),
                requires_grad=False,
            ).view(1, LIVE_VOCAB)
            hit = torch.ops.aten.full.default(
                [], -1.0, dtype=torch.float32, device=torch.device(device)
            )
            miss = torch.ops.aten.full.default(
                [], 0.0, dtype=torch.float32, device=torch.device(device)
            )
            selected = torch.ops.aten.where.self(target == iota, hit, miss)
            dense = torch.ops.aten.mul.Tensor(selected, scale)
            dense = torch.ops.prims.convert_element_type.default(
                dense, torch.bfloat16
            )
            dense = torch.ops.prims.convert_element_type.default(dense, torch.float32)
            row_sum = torch.ops.aten.sum.dim_IntList(dense, [1], True)
            return torch.ops.aten.sub.Tensor(dense, row_sum)

        target = torch.tensor([[0], [LIVE_VOCAB]], device=device)
        scale = torch.randn(2, 1, device=device)
        return make_fx(fn)(target, scale).graph

    @parametrize("force_shape_pad,expected_fold", ((False, True), (True, False)))
    @parametrize("bf16_roundtrip", (False, True))
    def test_cross_entropy_row_sum(
        self, device, force_shape_pad, expected_fold, bf16_roundtrip
    ):
        def fn(target, scale):
            iota = torch.arange(LIVE_VOCAB, device=target.device).view(
                1, LIVE_VOCAB
            )
            selected = torch.where(target == iota, -1.0, 0.0)
            dense = selected * scale
            if bf16_roundtrip:
                dense = dense.to(torch.bfloat16).to(torch.float32)
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense, dense - row_sum

        targets = [-100, -1, 0, LIVE_VOCAB - 1, LIVE_VOCAB, LIVE_VOCAB + 2]
        scales = [0.0, -0.0, 0.3333, float("inf"), -float("inf"), float("nan")]
        target = torch.tensor(targets, device=device).repeat_interleave(
            len(scales)
        )[:, None]
        scale = torch.tensor(scales, device=device).repeat(len(targets))[:, None]

        actual, expected = self._run_and_check(
            fn,
            (target, scale),
            expected_fold and bf16_roundtrip,
            {"force_shape_pad": force_shape_pad},
        )
        self.assertEqual(torch.signbit(actual[1]), torch.signbit(expected[1]))

    @parametrize(
        "variant", ("explicit_expand", "reversed_mask", "reversed_mul", "reshape")
    )
    def test_equivalent_graph_forms(self, device, variant):
        def fn(target, scale):
            iota = torch.arange(LIVE_VOCAB, device=target.device)
            iota = (
                iota.reshape(1, LIVE_VOCAB)
                if variant == "reshape"
                else iota.view(1, LIVE_VOCAB)
            )
            expanded = (
                target.expand(target.shape[0], LIVE_VOCAB)
                if variant == "explicit_expand"
                else target
            )
            mask = expanded == iota
            if variant == "reversed_mask":
                mask = iota == expanded
            selected = torch.where(mask, -1.0, 0.0)
            product = selected * scale
            if variant == "reversed_mul":
                product = scale * selected
            dense = product.to(torch.bfloat16).to(torch.float32)
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense - row_sum

        target = torch.tensor([[0], [LIVE_VOCAB - 1]], device=device)
        scale = torch.randn(2, 1, device=device)
        self._run_and_check(fn, (target, scale), True)

    def test_dynamic_batch(self, device):
        def fn(target, scale):
            iota = torch.arange(LIVE_VOCAB, device=target.device).view(
                1, LIVE_VOCAB
            )
            dense = torch.where(target == iota, -1.0, 0.0) * scale
            dense = dense.to(torch.bfloat16).to(torch.float32)
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense - row_sum

        inputs = []
        for batch in (3, 7):
            target = torch.randint(0, LIVE_VOCAB, (batch, 1), device=device)
            scale = torch.randn(batch, 1, device=device)
            inputs.append((target, scale))
        with config.patch(singleton_reduction_elimination=False):
            expected_fn = torch.compile(fn, fullgraph=True, dynamic=True)
            expected = [expected_fn(*args) for args in inputs]
        torch._dynamo.reset()
        counters.clear()
        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        for index, args in enumerate(inputs):
            actual = compiled(*args)
            self.assertEqual(actual, expected[index])
            if index == 0:
                self.assertEqual(
                    counters["inductor"]["singleton_reduction_elimination"], 1
                )

    @parametrize(
        "case",
        (
            "nonzero_miss",
            "negative_zero_miss",
            "varying_hit",
            "shifted_iota",
            "float16_rounding",
            "int32_index",
            "noncanonical_mul_hit",
            "second_mul",
            "extra_cast",
        ),
    )
    def test_rejects_unsupported_semantics(self, device, case):
        def fn(target, scale, varying):
            iota = torch.arange(
                LIVE_VOCAB,
                device=target.device,
                dtype=torch.int32 if case == "int32_index" else torch.int64,
            ).view(1, LIVE_VOCAB)
            if case == "shifted_iota":
                iota = iota + 1
            miss = -0.0 if case == "negative_zero_miss" else 0.0
            if case == "nonzero_miss":
                miss = 0.5
            if case == "varying_hit":
                hit = varying
            elif case == "noncanonical_mul_hit":
                hit = 2.0
            else:
                hit = -1.0
            dense = torch.where(target == iota, hit, miss) * scale
            if case == "second_mul":
                dense = dense * scale
            low_dtype = (
                torch.float16 if case == "float16_rounding" else torch.bfloat16
            )
            dense = dense.to(low_dtype).to(torch.float32)
            if case == "extra_cast":
                dense = dense.to(torch.bfloat16).to(torch.float32)
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense - row_sum

        index_dtype = torch.int32 if case == "int32_index" else torch.int64
        target = torch.tensor([[0], [LIVE_VOCAB]], device=device, dtype=index_dtype)
        scale = torch.randn(2, 1, device=device)
        varying = torch.randn(2, LIVE_VOCAB, device=device)
        self._run_and_check(fn, (target, scale, varying), False)

    @parametrize(
        "case",
        (
            "small_row",
            "no_expanding_user",
            "downstream_sum",
            "transitive_sum",
            "cat_sum",
            "split_sum",
        ),
    )
    def test_live_reuse_profitability(self, device, case):
        vocab = 128 if case == "small_row" else LIVE_VOCAB

        def fn(target, scale):
            iota = torch.arange(vocab, device=target.device).view(1, vocab)
            dense = torch.where(target == iota, -1.0, 0.0) * scale
            dense = dense.to(torch.bfloat16).to(torch.float32)
            row_sum = dense.sum(dim=1, keepdim=True)
            if case == "no_expanding_user":
                return dense, row_sum
            output = dense - row_sum
            if case == "downstream_sum":
                return output, dense.sum(dim=0, keepdim=True)
            if case == "transitive_sum":
                return output, (dense + 1).transpose(0, 1).sum(dim=1)
            if case == "cat_sum":
                return output, torch.cat((dense, dense), dim=0).sum(dim=0)
            if case == "split_sum":
                return output, torch.split(dense, (1, vocab - 1), dim=1)[0].sum()
            return output

        target = torch.tensor([[0], [vocab]], device=device)
        scale = torch.randn(2, 1, device=device)
        self._run_and_check(fn, (target, scale), False)

    def test_dynamic_reduction_extent(self, device):
        def fn(target, scale, dense):
            iota = torch.arange(dense.shape[1], device=dense.device).view(
                1, dense.shape[1]
            )
            selected = torch.where(target == iota, -1.0, 0.0) * scale
            selected = selected.to(torch.bfloat16).to(torch.float32)
            row_sum = selected.sum(dim=1, keepdim=True)
            return selected - row_sum

        target = torch.tensor([[0], [7]], device=device)
        scale = torch.randn(2, 1, device=device)
        dense = torch.randn(2, 8, device=device)
        torch._dynamo.mark_dynamic(dense, 1)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale, dense)
        torch._dynamo.reset()
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale, dense
        )
        self.assertEqual(actual, expected)
        self.assertEqual(counters["inductor"]["singleton_reduction_elimination"], 0)
        FileCheck().check("rnumel").run(code)

    def test_rank3_rejected(self, device):
        def fn(target, scale):
            iota = torch.arange(LIVE_VOCAB, device=target.device).view(
                1, 1, LIVE_VOCAB
            )
            dense = torch.where(target == iota, -1.0, 0.0) * scale
            dense = dense.to(torch.bfloat16).to(torch.float32)
            row_sum = dense.sum(dim=2, keepdim=True)
            return dense - row_sum

        target = torch.tensor([[[0]], [[LIVE_VOCAB]]], device=device)
        scale = torch.randn(2, 1, 1, device=device)
        self._run_and_check(fn, (target, scale), False)

    def test_analysis_limit(self, device):
        graph = self._make_pass_graph(device)
        with mock.patch.object(singleton_reduction, "_MAX_ANALYSIS_NODES", 1):
            self.assertEqual(eliminate_singleton_reductions(graph), 0)

    def test_downstream_reduction_analysis_is_cached(self, device):
        graph = self._make_pass_graph(device)
        reduction = next(
            node
            for node in graph.nodes
            if node.target is torch.ops.aten.sum.dim_IntList
        )
        consumer = next(iter(reduction.users))
        output = next(node for node in graph.nodes if node.op == "output")
        with graph.inserting_before(output):
            extra_reduction = graph.node_copy(reduction, lambda node: node)
            graph.node_copy(
                consumer,
                lambda node: extra_reduction if node is reduction else node,
            )

        find_reductions = singleton_reduction._find_downstream_reductions
        with mock.patch.object(
            singleton_reduction,
            "_find_downstream_reductions",
            wraps=find_reductions,
        ) as find:
            self.assertEqual(eliminate_singleton_reductions(graph), 0)
        self.assertEqual(find.call_count, 1)

    def test_unknown_downstream_node_rejected(self, device):
        def opaque(value):
            return value

        graph = self._make_pass_graph(device)
        reduction = next(
            node
            for node in graph.nodes
            if node.target is torch.ops.aten.sum.dim_IntList
        )
        dense = reduction.args[0]
        output = next(node for node in graph.nodes if node.op == "output")
        with graph.inserting_before(output):
            graph.call_function(opaque, (dense,))

        self.assertEqual(eliminate_singleton_reductions(graph), 0)

    def test_hip_rejected(self, device):
        graph = self._make_pass_graph(device)
        with mock.patch.object(torch.version, "hip", "test"):
            self.assertEqual(eliminate_singleton_reductions(graph), 0)

    def test_does_not_run_global_dce(self, device):
        graph = self._make_pass_graph(device)
        placeholder = next(node for node in graph.nodes if node.op == "placeholder")
        output = next(node for node in graph.nodes if node.op == "output")
        with graph.inserting_before(output):
            dead = graph.call_function(torch.ops.aten.neg.default, (placeholder,))

        self.assertEqual(eliminate_singleton_reductions(graph), 1)
        self.assertIn(dead, graph.nodes)


devices = ("cuda",) if TEST_CUDA else ()
instantiate_device_type_tests(
    SingletonReductionTests,
    globals(),
    only_for=devices,
)


if __name__ == "__main__":
    run_tests()
