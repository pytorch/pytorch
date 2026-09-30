# Owner(s): ["module: inductor"]

"""End-to-end coverage for opt-in translated staged-reduction fusion."""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend
from torch._inductor import metrics
import torch._inductor.config as inductor_config
from torch._inductor.choices import InductorChoices
from torch._inductor.dependencies import MemoryDep
from torch._inductor.scheduler import (
    FusedNestedReductions,
    FusedStagedReduction,
    NestedReduction,
    Scheduler,
)
from torch._inductor.test_case import TestCase, run_tests
from torch._inductor.utils import fresh_inductor_cache, sympy_index_symbol
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize


HEAD_DIM = 256
QK_ROPE_A = 64
HIDDEN_SIZE = 7168


def shifted_mla_indexer(x, ln_w, ln_b, cos, sin):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    normed = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b
    k_rot, k_pass = torch.split(
        normed.unsqueeze(2),
        [QK_ROPE_A, x.shape[-1] - QK_ROPE_A],
        dim=-1,
    )
    return k_rot * cos + k_rot * sin, k_pass


def flinear_shifted_mla_indexer(hidden_states, wk_weight, ln_w, ln_b, cos, sin):
    projected = F.linear(hidden_states, wk_weight)
    return shifted_mla_indexer(projected, ln_w, ln_b, cos, sin)


def strided_shifted_mla_indexer(x, ln_w, ln_b, cos, sin):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    normed = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b
    strided = normed[..., ::2].unsqueeze(2)
    leading = strided[..., :32]
    return leading * cos[..., :32] + leading * sin[..., :32], strided[..., 32:]


def indirect_shifted_mla_indexer(x, ln_w, ln_b, cos, sin):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    normed = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b
    indices = torch.arange(96, device=x.device) * 2
    gathered = normed.index_select(-1, indices).unsqueeze(2)
    leading = gathered[..., :32]
    return leading * cos[..., :32] + leading * sin[..., :32], gathered[..., 32:]


def equal_split_mla_indexer(x, ln_w, ln_b):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    normed = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b
    return torch.split(normed.unsqueeze(2), [HEAD_DIM // 2, HEAD_DIM // 2], dim=-1)


def _make_mla_inputs(
    *, device, batch_size: int, seq_len: int, head_dim: int = HEAD_DIM
):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return (
        torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype),
        torch.randn(head_dim, device=device, dtype=dtype),
        torch.randn(head_dim, device=device, dtype=dtype),
        torch.randn(
            batch_size, seq_len, 1, QK_ROPE_A, device=device, dtype=dtype
        ),
        torch.randn(
            batch_size, seq_len, 1, QK_ROPE_A, device=device, dtype=dtype
        ),
    )


def _make_flinear_inputs(*, device, batch_size: int, seq_len: int):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return (
        torch.randn(
            batch_size,
            seq_len,
            HIDDEN_SIZE,
            device=device,
            dtype=dtype,
        ),
        torch.randn(HEAD_DIM, HIDDEN_SIZE, device=device, dtype=dtype),
        torch.randn(HEAD_DIM, device=device, dtype=dtype),
        torch.randn(HEAD_DIM, device=device, dtype=dtype),
        torch.randn(
            batch_size, seq_len, 1, QK_ROPE_A, device=device, dtype=dtype
        ),
        torch.randn(
            batch_size, seq_len, 1, QK_ROPE_A, device=device, dtype=dtype
        ),
    )


@dataclasses.dataclass(frozen=True)
class _Observation:
    outputs: tuple[torch.Tensor, ...]
    generated_kernel_count: int
    staged_fusion_count: int
    affine_mappings: tuple[tuple[int, int, int], ...]
    logical_factors: tuple[int, ...]
    output_group_affine_mappings: tuple[tuple[int, int, int], ...]
    parent_widths: tuple[int, ...] = ()
    lifetime_signatures: tuple[tuple[str, int, int, int], ...] = ()


def _capture_staged_plans(nodes, staged_plans, output_group_mappings):
    for node in nodes:
        if not isinstance(node, FusedStagedReduction) or isinstance(
            node, FusedNestedReductions
        ):
            continue
        device = node.get_device()
        if device is None:
            continue
        _, (numel, rnumel) = node.group
        plan = NestedReduction.sub_parent_epilogue_plan(node.get_nodes(), numel, rnumel)
        if plan is not None:
            staged_plans.append(plan)
            output_group_mappings.extend(
                _plan_affine_mappings([plan], output_groups_only=True)
            )


def _plan_affine_mappings(staged_plans, *, output_groups_only=False):
    def relation_matches_group(relation, plan, extent, group):
        frame = (
            sympy_index_symbol("_sub_parent_replay_x"),
            sympy_index_symbol("_sub_parent_replay_r"),
        )
        sizes = (plan.parent_numel, extent)
        for node in group.nodes:
            for read in node.read_writes.reads:
                if (
                    not isinstance(read, MemoryDep)
                    or relation.consumer_access.name != read.name
                    or relation.consumer_access.mode != read.mode
                ):
                    continue
                left = relation.consumer_access.normalize_with_ranges(frame, sizes)
                right = read.normalize_with_ranges(frame, sizes)
                if (
                    left is not None
                    and right is not None
                    and V.graph.sizevars.statically_known_equals(
                        left.index, right.index
                    )
                ):
                    return True
        return False

    return tuple(
        (
            relation.access_stride,
            relation.base_offset,
            relation.extent,
        )
        for plan in staged_plans
        for stage in plan.sub_parent_stages
        for relation in stage.access_relations
        if relation.access_stride is not None
        and (
            not output_groups_only
            or any(
                relation_matches_group(
                    relation, plan, relation.extent, group
                )
                for group in stage.output_groups
            )
        )
    )


def _choices_context(force_persistent: bool | None):
    if force_persistent is None:
        return nullcontext()

    class _Choices(InductorChoices):
        @staticmethod
        def should_use_cooperative_reduction(*args, **kwargs):
            return False

        @staticmethod
        def should_use_persistent_reduction(*args, **kwargs):
            return force_persistent

    return V.set_choices_handler(_Choices())


def _observe(
    fn,
    inputs: tuple[torch.Tensor, ...],
    *,
    polyhedral_fusion: bool,
    force_persistent: bool | None = None,
    nested_reduction: bool = True,
    reorder_for_peak_memory: bool | None = None,
    memory_planning: bool | None = None,
    memory_pool: str | None = None,
) -> _Observation:
    torch._dynamo.reset()
    metrics.reset()
    staged_plans = []
    output_group_mappings = []
    lifetime_signatures = []
    original_compute_last_usage = Scheduler.compute_last_usage

    def capture(nodes):
        _capture_staged_plans(nodes, staged_plans, output_group_mappings)
        return nodes

    def capture_last_usage(scheduler):
        original_compute_last_usage(scheduler)
        lifetime_signatures.extend(
            (
                type(node).__name__,
                len(node.get_nodes()),
                len(node.get_outputs()),
                len(node.last_usage),
            )
            for node in scheduler.nodes
        )

    compile_config = {
        "polyhedral_fusion": polyhedral_fusion,
        "_post_fusion_custom_pass": capture,
        "fx_graph_cache": False,
    }
    if reorder_for_peak_memory is not None:
        compile_config["reorder_for_peak_memory"] = reorder_for_peak_memory
    if memory_planning is not None:
        compile_config["memory_planning"] = memory_planning
    if memory_pool is not None:
        compile_config["memory_pool"] = memory_pool

    with (
        inductor_config.patch(compile_config),
        inductor_config.patch("triton.nested_reduction", nested_reduction),
        fresh_inductor_cache(),
        _choices_context(force_persistent),
        patch.object(Scheduler, "compute_last_usage", capture_last_usage),
    ):
        compiled = torch.compile(fn, fullgraph=True)
        outputs = compiled(*inputs)

    if not isinstance(outputs, tuple):
        raise AssertionError("translated MLA fixture must return a tuple")
    affine_mappings = _plan_affine_mappings(staged_plans)
    output_group_affine_mappings = tuple(output_group_mappings)
    logical_factors = tuple(
        stage.factor
        for plan in staged_plans
        for stage in plan.sub_parent_stages
    )
    parent_widths = tuple(int(plan.parent_rnumel) for plan in staged_plans)
    return _Observation(
        outputs=tuple(outputs),
        generated_kernel_count=metrics.generated_kernel_count,
        staged_fusion_count=len(staged_plans),
        affine_mappings=affine_mappings,
        output_group_affine_mappings=output_group_affine_mappings,
        logical_factors=logical_factors,
        parent_widths=parent_widths,
        lifetime_signatures=tuple(lifetime_signatures),
    )


def _mark_dynamic_batch_sequence(
    inputs: tuple[torch.Tensor, ...], *, dynamic_feature_width: bool = False
) -> None:
    for input_index, tensor in enumerate(inputs):
        for dim in range(tensor.dim()):
            is_feature_width = dynamic_feature_width and (
                (input_index == 0 and dim == 2)
                or (input_index in (1, 2) and dim == 0)
            )
            if is_feature_width:
                # Keep this dimension symbolic while testing dynamic-width fallback.
                torch._dynamo.maybe_mark_dynamic(tensor, dim)
            elif tensor.dim() >= 2 and dim < 2:
                torch._dynamo.mark_dynamic(tensor, dim)
            else:
                torch._dynamo.mark_static(tensor, dim)


def _observe_dynamic(
    fn,
    inputs_by_shape: tuple[tuple[torch.Tensor, ...], ...],
    *,
    polyhedral_fusion: bool,
    dynamic_feature_width: bool = False,
    backend=None,
):
    torch._dynamo.reset()
    metrics.reset()
    staged_plans = []
    output_group_mappings = []

    def capture(nodes):
        _capture_staged_plans(nodes, staged_plans, output_group_mappings)
        return nodes

    _mark_dynamic_batch_sequence(
        inputs_by_shape[0], dynamic_feature_width=dynamic_feature_width
    )
    outputs = []
    with (
        inductor_config.patch(
            polyhedral_fusion=polyhedral_fusion,
            _post_fusion_custom_pass=capture,
            fx_graph_cache=False,
        ),
        inductor_config.patch("triton.nested_reduction", True),
        fresh_inductor_cache(),
    ):
        if backend is None:
            compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        else:
            compiled = torch.compile(
                fn, backend=backend, fullgraph=True, dynamic=True
            )
        for inputs in inputs_by_shape:
            result = compiled(*inputs)
            if not isinstance(result, tuple):
                raise AssertionError("dynamic MLA fixture must return a tuple")
            outputs.append(tuple(result))

    affine_mappings = _plan_affine_mappings(staged_plans)
    output_group_affine_mappings = tuple(output_group_mappings)
    logical_factors = tuple(
        stage.factor
        for plan in staged_plans
        for stage in plan.sub_parent_stages
    )
    parent_widths = tuple(int(plan.parent_rnumel) for plan in staged_plans)
    return outputs, _Observation(
        outputs=tuple(outputs[-1]),
        generated_kernel_count=metrics.generated_kernel_count,
        staged_fusion_count=len(staged_plans),
        affine_mappings=affine_mappings,
        output_group_affine_mappings=output_group_affine_mappings,
        logical_factors=logical_factors,
        parent_widths=parent_widths,
    )


class PolyhedralMLAFusionTest(TestCase):
    def assert_outputs(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for result, reference in zip(actual, expected):
            self.assertEqual(result.shape, reference.shape)
            self.assertEqual(result.dtype, reference.dtype)
            self.assertTrue(torch.isfinite(result).all().item())
        self.assertEqual(actual, expected, atol=6e-2, rtol=2e-2)

    def assert_affine_plan(self, observation: _Observation) -> None:
        self.assertGreaterEqual(observation.staged_fusion_count, 1)
        self.assertEqual(
            set(observation.affine_mappings),
            {(1, 0, QK_ROPE_A), (1, QK_ROPE_A, HEAD_DIM - QK_ROPE_A)},
        )
        self.assertEqual(
            set(observation.output_group_affine_mappings), {(1, 0, QK_ROPE_A)}
        )
        self.assertEqual(set(observation.logical_factors), {4})
        self.assertEqual(set(observation.parent_widths), {HEAD_DIM})

    @parametrize("force_persistent", [False, True])
    @parametrize("view_offset", [1, 64])
    def test_existing_lane_fusion_with_output_view(
        self, device, force_persistent, view_offset
    ):
        def fn(x):
            y = x + x.sum(-1, keepdim=True)
            return y[:, ::4] * 2, y[:, view_offset:]

        torch.manual_seed(0)
        inputs = (torch.randn(8, 128, device=device),)
        eager = fn(inputs[0])
        disabled = _observe(
            fn, inputs, polyhedral_fusion=False, force_persistent=force_persistent
        )
        enabled = _observe(
            fn, inputs, polyhedral_fusion=True, force_persistent=force_persistent
        )
        self.assert_outputs(disabled.outputs, eager)
        self.assert_outputs(enabled.outputs, eager)
        self.assertEqual(disabled.staged_fusion_count, 1)
        self.assertEqual(enabled.staged_fusion_count, disabled.staged_fusion_count)
        self.assertEqual(disabled.generated_kernel_count, 1)
        self.assertEqual(enabled.generated_kernel_count, disabled.generated_kernel_count)
        self.assertEqual(enabled.affine_mappings, disabled.affine_mappings)

    @parametrize(
        "fn,shape_name,batch_size,seq_len,reorder,memory_planning",
        (
            (shifted_mla_indexer, "smoke", 2, 8, False, False),
            (shifted_mla_indexer, "prefill", 4, 512, True, False),
            (shifted_mla_indexer, "decode", 64, 1, True, True),
            (flinear_shifted_mla_indexer, "projection", 2, 8, False, False),
        ),
    )
    def test_translated_fusion(
        self, device, fn, shape_name, batch_size, seq_len, reorder, memory_planning
    ):
        make_inputs = (
            _make_flinear_inputs if fn is flinear_shifted_mla_indexer else _make_mla_inputs
        )
        inputs = make_inputs(device=device, batch_size=batch_size, seq_len=seq_len)
        eager = tuple(fn(*inputs))
        disabled, enabled = (
            _observe(
                fn,
                inputs,
                polyhedral_fusion=enabled,
                force_persistent=True,
                reorder_for_peak_memory=reorder,
                memory_planning=memory_planning,
                memory_pool="intermediates" if memory_planning else "none",
            )
            for enabled in (False, True)
        )
        self.assert_outputs(disabled.outputs, eager)
        self.assert_outputs(enabled.outputs, eager)
        self.assertEqual(disabled.staged_fusion_count, 0)
        self.assert_affine_plan(enabled)
        if shape_name == "smoke":
            self.assertLess(
                enabled.generated_kernel_count, disabled.generated_kernel_count
            )
        staged_lifetimes = [
            signature
            for signature in enabled.lifetime_signatures
            if signature[0] == "FusedStagedReduction"
        ]
        self.assertEqual(len(staged_lifetimes), enabled.staged_fusion_count)
        for _, inner_nodes, outputs, last_usage in staged_lifetimes:
            self.assertGreater(inner_nodes, 1)
            self.assertGreater(outputs, 0)
            self.assertGreater(last_usage, 0)

    @parametrize(
        "dynamic_feature_width,shapes",
        (
            (False, ((2, 8, 256), (4, 5, 256), (64, 1, 256))),
            (True, ((2, 8, 256), (2, 8, 192), (2, 8, 384))),
        ),
    )
    def test_dynamic_shapes(self, device, dynamic_feature_width, shapes):
        inputs_by_shape = tuple(
            _make_mla_inputs(
                device=device, batch_size=batch_size, seq_len=seq_len, head_dim=head_dim
            )
            for batch_size, seq_len, head_dim in shapes
        )
        eager = [tuple(shifted_mla_indexer(*inputs)) for inputs in inputs_by_shape]
        counters = [CompileCounterWithBackend("inductor") for _ in range(2)]
        (disabled_outputs, disabled), (enabled_outputs, enabled) = (
            _observe_dynamic(
                shifted_mla_indexer,
                inputs_by_shape,
                polyhedral_fusion=enabled,
                dynamic_feature_width=dynamic_feature_width,
                backend=counter,
            )
            for enabled, counter in zip((False, True), counters)
        )
        for expected, disabled_result, enabled_result in zip(
            eager, disabled_outputs, enabled_outputs
        ):
            self.assert_outputs(disabled_result, expected)
            self.assert_outputs(enabled_result, expected)
        self.assertEqual(disabled.staged_fusion_count, 0)
        if dynamic_feature_width:
            self.assertEqual(enabled.staged_fusion_count, 0)
            for counter in counters:
                self.assertEqual(counter.frame_count, 1)
        else:
            self.assert_affine_plan(enabled)

    @parametrize(
        "fn,head_dim,options",
        (
            (shifted_mla_indexer, 192, {}),
            (shifted_mla_indexer, 384, {}),
            (strided_shifted_mla_indexer, HEAD_DIM, {}),
            (indirect_shifted_mla_indexer, HEAD_DIM, {}),
            (equal_split_mla_indexer, HEAD_DIM, {}),
            (shifted_mla_indexer, HEAD_DIM, {"nested_reduction": False}),
            (shifted_mla_indexer, HEAD_DIM, {"force_persistent": False}),
        ),
    )
    def test_unsupported_candidate_falls_back(self, device, fn, head_dim, options):
        inputs = _make_mla_inputs(device=device, batch_size=2, seq_len=8, head_dim=head_dim)
        if fn is equal_split_mla_indexer:
            inputs = inputs[:3]
        observation = _observe(fn, inputs, polyhedral_fusion=True, **options)
        self.assert_outputs(observation.outputs, tuple(fn(*inputs)))
        self.assertEqual(observation.staged_fusion_count, 0)


instantiate_device_type_tests(PolyhedralMLAFusionTest, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
