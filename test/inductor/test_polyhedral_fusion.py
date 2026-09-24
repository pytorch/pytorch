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
from torch._inductor.scheduler import (
    FusedNestedReductions,
    FusedStagedReduction,
    NestedReduction,
    Scheduler,
)
from torch._inductor.test_case import TestCase, run_tests
from torch._inductor.utils import fresh_inductor_cache
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


HEAD_DIM = 256
QK_ROPE_A = 64
HIDDEN_SIZE = 7168


def _shifted_mla_indexer(x, ln_w, ln_b, cos, sin, rope_width):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    normed = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b
    k_rot, k_pass = torch.split(
        normed.unsqueeze(2),
        [rope_width, x.shape[-1] - rope_width],
        dim=-1,
    )
    return k_rot * cos + k_rot * sin, k_pass


def shifted_mla_indexer(x, ln_w, ln_b, cos, sin):
    return _shifted_mla_indexer(x, ln_w, ln_b, cos, sin, QK_ROPE_A)


def flinear_shifted_mla_indexer(hidden_states, wk_weight, ln_w, ln_b, cos, sin):
    projected = F.linear(hidden_states, wk_weight)
    return shifted_mla_indexer(projected, ln_w, ln_b, cos, sin)


def shifted_mla_external_consumer(x, ln_w, ln_b, cos, sin):
    k_rot, k_pass = shifted_mla_indexer(x, ln_w, ln_b, cos, sin)
    return k_rot, k_pass, k_pass.sin()


def shifted_mla_mutating_indexer(x, ln_w, ln_b, cos, sin, output):
    k_rot, k_pass = shifted_mla_indexer(x, ln_w, ln_b, cos, sin)
    output.copy_(k_pass)
    return k_rot, k_pass, output


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


def stride_three_norm(x, ln_w, ln_b, scale):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    normed = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b
    return (normed[..., ::3] * scale,)


def equal_split_mla_indexer(x, ln_w, ln_b):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    normed = (x - mean) / torch.sqrt(var + 1e-5) * ln_w + ln_b
    return torch.split(normed.unsqueeze(2), [HEAD_DIM // 2, HEAD_DIM // 2], dim=-1)


def _make_mla_inputs(
    *, batch_size: int, seq_len: int, head_dim: int = HEAD_DIM, seed: int = 0
):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    return (
        torch.randn(batch_size, seq_len, head_dim, device=GPU_TYPE, dtype=dtype),
        torch.randn(head_dim, device=GPU_TYPE, dtype=dtype),
        torch.randn(head_dim, device=GPU_TYPE, dtype=dtype),
        torch.randn(
            batch_size, seq_len, 1, QK_ROPE_A, device=GPU_TYPE, dtype=dtype
        ),
        torch.randn(
            batch_size, seq_len, 1, QK_ROPE_A, device=GPU_TYPE, dtype=dtype
        ),
    )


def _make_flinear_inputs(*, batch_size: int, seq_len: int):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return (
        torch.randn(
            batch_size,
            seq_len,
            HIDDEN_SIZE,
            device=GPU_TYPE,
            dtype=dtype,
        ),
        torch.randn(HEAD_DIM, HIDDEN_SIZE, device=GPU_TYPE, dtype=dtype),
        torch.randn(HEAD_DIM, device=GPU_TYPE, dtype=dtype),
        torch.randn(HEAD_DIM, device=GPU_TYPE, dtype=dtype),
        torch.randn(
            batch_size, seq_len, 1, QK_ROPE_A, device=GPU_TYPE, dtype=dtype
        ),
        torch.randn(
            batch_size, seq_len, 1, QK_ROPE_A, device=GPU_TYPE, dtype=dtype
        ),
    )


@dataclasses.dataclass(frozen=True)
class _Observation:
    outputs: tuple[torch.Tensor, ...]
    generated_kernel_count: int
    staged_fusion_count: int
    translations: tuple[tuple[object, ...], ...]
    logical_factors: tuple[int, ...]
    parent_widths: tuple[int, ...] = ()
    lifetime_signatures: tuple[tuple[str, int, int, int], ...] = ()


def _capture_staged_plans(nodes, staged_plans):
    for node in nodes:
        if not isinstance(node, FusedStagedReduction) or isinstance(
            node, FusedNestedReductions
        ):
            continue
        reductions = [
            candidate for candidate in node.get_nodes() if candidate.is_reduction()
        ]
        if not reductions:
            continue
        _, (parent_numel, parent_rnumel) = reductions[0].group
        plan = NestedReduction.sub_parent_epilogue_plan(
            node.get_nodes(), parent_numel, parent_rnumel
        )
        if plan is not None:
            staged_plans.append(plan)


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
    lifetime_signatures = []
    original_compute_last_usage = Scheduler.compute_last_usage

    def capture(nodes):
        _capture_staged_plans(nodes, staged_plans)
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
    translations = tuple(
        relation.translation
        for plan in staged_plans
        for stage in plan.sub_parent_stages
        for relation in stage.access_relations
    )
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
        translations=translations,
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

    def capture(nodes):
        _capture_staged_plans(nodes, staged_plans)
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

    translations = tuple(
        relation.translation
        for plan in staged_plans
        for stage in plan.sub_parent_stages
        for relation in stage.access_relations
    )
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
        translations=translations,
        logical_factors=logical_factors,
        parent_widths=parent_widths,
    )


class PolyhedralMLAFusionTest(TestCase):
    __unittest_skip__ = not HAS_GPU

    def assert_translated_plan(self, observation: _Observation) -> None:
        self.assertGreaterEqual(observation.staged_fusion_count, 1)
        self.assertEqual(set(observation.translations), {(0, 0), (0, QK_ROPE_A)})

    def test_static_shape_matrix(self):
        for shape_name, batch_size, seq_len in (
            ("smoke", 2, 8),
            ("prefill", 4, 512),
            ("decode", 64, 1),
        ):
            with self.subTest(shape=shape_name):
                inputs = _make_mla_inputs(batch_size=batch_size, seq_len=seq_len)
                eager = tuple(shifted_mla_indexer(*inputs))
                disabled = _observe(
                    shifted_mla_indexer,
                    inputs,
                    polyhedral_fusion=False,
                )
                enabled = _observe(
                    shifted_mla_indexer,
                    inputs,
                    polyhedral_fusion=True,
                )
                self.assertEqual(
                    disabled.outputs, eager, atol=6e-2, rtol=2e-2
                )
                self.assertEqual(
                    enabled.outputs, eager, atol=6e-2, rtol=2e-2
                )
                self.assertEqual(disabled.staged_fusion_count, 0)
                self.assert_translated_plan(enabled)
                if shape_name == "smoke":
                    self.assertLess(
                        enabled.generated_kernel_count,
                        disabled.generated_kernel_count,
                    )
    @parametrize("force_persistent", (True, False))
    def test_looped_and_persistent_translated_fusion(self, force_persistent):
        inputs = _make_mla_inputs(batch_size=2, seq_len=8)
        eager = tuple(shifted_mla_indexer(*inputs))
        with self.subTest(force_persistent=force_persistent):
            disabled = _observe(
                shifted_mla_indexer,
                inputs,
                polyhedral_fusion=False,
                force_persistent=force_persistent,
            )
            enabled = _observe(
                shifted_mla_indexer,
                inputs,
                polyhedral_fusion=True,
                force_persistent=force_persistent,
            )
            self.assertEqual(
                disabled.outputs,
                eager,
                atol=6e-2,
                rtol=2e-2,
            )
            self.assertEqual(
                enabled.outputs,
                eager,
                atol=6e-2,
                rtol=2e-2,
            )
            self.assertEqual(disabled.staged_fusion_count, 0)
            if force_persistent:
                self.assert_translated_plan(enabled)
            else:
                self.assertEqual(enabled.staged_fusion_count, 0)


    def test_nested_reduction_gate(self):
        inputs = _make_mla_inputs(batch_size=2, seq_len=8)
        eager = tuple(shifted_mla_indexer(*inputs))
        observation = _observe(
            shifted_mla_indexer,
            inputs,
            polyhedral_fusion=True,
            nested_reduction=False,
        )
        self.assertEqual(
            observation.outputs, eager, atol=6e-2, rtol=2e-2
        )
        self.assertEqual(observation.staged_fusion_count, 0)
        self.assertEqual(observation.translations, ())

    def test_dynamic_batch_and_sequence(self):
        inputs_by_shape = tuple(
            _make_mla_inputs(batch_size=batch_size, seq_len=seq_len)
            for batch_size, seq_len in ((2, 8), (4, 5), (64, 1))
        )
        eager = [tuple(shifted_mla_indexer(*inputs)) for inputs in inputs_by_shape]
        disabled_outputs, disabled = _observe_dynamic(
            shifted_mla_indexer,
            inputs_by_shape,
            polyhedral_fusion=False,
        )
        enabled_outputs, enabled = _observe_dynamic(
            shifted_mla_indexer,
            inputs_by_shape,
            polyhedral_fusion=True,
        )
        for expected, disabled_result, enabled_result in zip(
            eager, disabled_outputs, enabled_outputs
        ):
            self.assertEqual(
                disabled_result, expected, atol=6e-2, rtol=2e-2
            )
            self.assertEqual(
                enabled_result, expected, atol=6e-2, rtol=2e-2
            )
        self.assert_translated_plan(enabled)

    def test_dynamic_feature_width_falls_back(self):
        inputs_by_shape = tuple(
            _make_mla_inputs(
                batch_size=batch_size,
                seq_len=seq_len,
                head_dim=head_dim,
            )
            for batch_size, seq_len, head_dim in (
                (2, 8, 256),
                (2, 8, 192),
                (2, 8, 384),
                (4, 5, 192),
                (64, 1, 256),
            )
        )
        eager = [tuple(shifted_mla_indexer(*inputs)) for inputs in inputs_by_shape]
        disabled_outputs, disabled = _observe_dynamic(
            shifted_mla_indexer,
            inputs_by_shape,
            polyhedral_fusion=False,
            dynamic_feature_width=True,
        )
        enabled_outputs, enabled = _observe_dynamic(
            shifted_mla_indexer,
            inputs_by_shape,
            polyhedral_fusion=True,
            dynamic_feature_width=True,
        )
        for expected, disabled_result, enabled_result in zip(
            eager, disabled_outputs, enabled_outputs
        ):
            self.assertEqual(
                disabled_result, expected, atol=6e-2, rtol=2e-2
            )
            self.assertEqual(
                enabled_result, expected, atol=6e-2, rtol=2e-2
            )

        self.assertEqual(disabled.staged_fusion_count, 0)
        self.assertEqual(enabled.staged_fusion_count, 0)
        self.assertEqual(disabled.translations, ())
        self.assertEqual(enabled.translations, ())

    def test_dynamic_unsupported_width_reuses_fallback_graph(self):
        inputs_by_shape = tuple(
            _make_mla_inputs(batch_size=2, seq_len=8, head_dim=head_dim)
            for head_dim in (192, 384)
        )
        eager = [tuple(shifted_mla_indexer(*inputs)) for inputs in inputs_by_shape]
        counter = CompileCounterWithBackend("inductor")
        outputs, observation = _observe_dynamic(
            shifted_mla_indexer,
            inputs_by_shape,
            polyhedral_fusion=True,
            dynamic_feature_width=True,
            backend=counter,
        )

        for expected, result in zip(eager, outputs):
            self.assertEqual(result, expected, atol=6e-2, rtol=2e-2)
        self.assertEqual(observation.staged_fusion_count, 0)
        self.assertEqual(counter.frame_count, 1)

    def test_dynamic_rejected_power_of_two_width_reuses_fallback_graph(self):
        inputs_by_shape = tuple(
            _make_mla_inputs(batch_size=2, seq_len=8, head_dim=head_dim)
            for head_dim in (256, 512)
        )
        eager = [
            tuple(strided_shifted_mla_indexer(*inputs))
            for inputs in inputs_by_shape
        ]
        disabled_counter = CompileCounterWithBackend("inductor")
        disabled_outputs, disabled = _observe_dynamic(
            strided_shifted_mla_indexer,
            inputs_by_shape,
            polyhedral_fusion=False,
            dynamic_feature_width=True,
            backend=disabled_counter,
        )
        enabled_counter = CompileCounterWithBackend("inductor")
        enabled_outputs, enabled = _observe_dynamic(
            strided_shifted_mla_indexer,
            inputs_by_shape,
            polyhedral_fusion=True,
            dynamic_feature_width=True,
            backend=enabled_counter,
        )

        for expected, disabled_result, enabled_result in zip(
            eager, disabled_outputs, enabled_outputs
        ):
            self.assertEqual(
                disabled_result, expected, atol=6e-2, rtol=2e-2
            )
            self.assertEqual(
                enabled_result, expected, atol=6e-2, rtol=2e-2
            )
        self.assertEqual(disabled.staged_fusion_count, 0)
        self.assertEqual(enabled.staged_fusion_count, 0)
        self.assertEqual(disabled.translations, ())
        self.assertEqual(enabled.translations, ())
        self.assertEqual(disabled_counter.frame_count, 1)
        self.assertEqual(enabled_counter.frame_count, disabled_counter.frame_count)

    def test_wider_logical_factor_falls_back(self):
        inputs = _make_mla_inputs(batch_size=2, seq_len=8, head_dim=384)
        eager = tuple(shifted_mla_indexer(*inputs))
        disabled = _observe(
            shifted_mla_indexer,
            inputs,
            polyhedral_fusion=False,
            force_persistent=True,
        )
        enabled = _observe(
            shifted_mla_indexer,
            inputs,
            polyhedral_fusion=True,
            force_persistent=True,
        )
        self.assertEqual(
            disabled.outputs, eager, atol=6e-2, rtol=2e-2
        )
        self.assertEqual(
            enabled.outputs, eager, atol=6e-2, rtol=2e-2
        )
        self.assertEqual(enabled.staged_fusion_count, 0)
        self.assertEqual(enabled.translations, ())

    def test_non_power_of_two_parent_width_falls_back(self):
        inputs = _make_mla_inputs(batch_size=2, seq_len=8, head_dim=192)
        eager = tuple(shifted_mla_indexer(*inputs))
        observation = _observe(
            shifted_mla_indexer,
            inputs,
            polyhedral_fusion=True,
        )
        self.assertEqual(
            observation.outputs, eager, atol=6e-2, rtol=2e-2
        )
        self.assertEqual(observation.staged_fusion_count, 0)
        self.assertEqual(observation.translations, ())

    def test_non_power_of_two_rate_falls_back(self):
        base_inputs = _make_mla_inputs(batch_size=2, seq_len=8, head_dim=192)
        inputs = (*base_inputs[:3], base_inputs[3].squeeze(2))
        eager = tuple(stride_three_norm(*inputs))
        disabled = _observe(
            stride_three_norm,
            inputs,
            polyhedral_fusion=False,
            force_persistent=True,
        )
        enabled = _observe(
            stride_three_norm,
            inputs,
            polyhedral_fusion=True,
            force_persistent=True,
        )
        self.assertEqual(disabled.outputs, eager, atol=6e-2, rtol=2e-2)
        self.assertEqual(enabled.outputs, eager, atol=6e-2, rtol=2e-2)
        self.assertEqual(disabled.staged_fusion_count, 0)
        self.assertEqual(enabled.staged_fusion_count, 0)
        self.assertEqual(enabled.translations, ())

    def test_flinear_boundary(self):
        inputs = _make_flinear_inputs(batch_size=2, seq_len=8)
        eager = tuple(flinear_shifted_mla_indexer(*inputs))
        disabled = _observe(
            flinear_shifted_mla_indexer,
            inputs,
            polyhedral_fusion=False,
        )
        enabled = _observe(
            flinear_shifted_mla_indexer,
            inputs,
            polyhedral_fusion=True,
        )
        self.assertEqual(
            disabled.outputs, eager, atol=6e-2, rtol=2e-2
        )
        self.assertEqual(
            enabled.outputs, eager, atol=6e-2, rtol=2e-2
        )
        self.assert_translated_plan(enabled)

    def test_translated_lifetime_survives_external_consumer(self):
        inputs = _make_mla_inputs(batch_size=2, seq_len=8)
        eager = tuple(shifted_mla_external_consumer(*inputs))
        observation = _observe(
            shifted_mla_external_consumer,
            inputs,
            polyhedral_fusion=True,
        )
        self.assertEqual(
            observation.outputs, eager, atol=6e-2, rtol=2e-2
        )
        self.assert_translated_plan(observation)
        self.assertEqual(
            sum(
                signature[0] == "FusedStagedReduction"
                for signature in observation.lifetime_signatures
            ),
            observation.staged_fusion_count,
        )
        self.assertTrue(
            any(
                signature[0] == "SchedulerNode"
                for signature in observation.lifetime_signatures
            )
        )

    def test_translated_lifetime_survives_mutation(self):
        inputs = _make_mla_inputs(batch_size=2, seq_len=8)
        eager_output = torch.empty(
            2, 8, 1, 192, device=GPU_TYPE, dtype=torch.bfloat16
        )
        eager = tuple(
            shifted_mla_mutating_indexer(
                *inputs,
                eager_output,
            )
        )

        disabled_output = torch.empty_like(eager_output)
        disabled = _observe(
            shifted_mla_mutating_indexer,
            (*inputs, disabled_output),
            polyhedral_fusion=False,
        )
        enabled_output = torch.empty_like(eager_output)
        enabled = _observe(
            shifted_mla_mutating_indexer,
            (*inputs, enabled_output),
            polyhedral_fusion=True,
        )
        self.assertEqual(
            disabled.outputs, eager, atol=6e-2, rtol=2e-2
        )
        self.assertEqual(
            enabled.outputs, eager, atol=6e-2, rtol=2e-2
        )
        self.assert_translated_plan(enabled)
        self.assertTrue(
            all(
                inner_nodes > 1 and outputs > 0 and last_usage > 0
                for kind, inner_nodes, outputs, last_usage in enabled.lifetime_signatures
                if kind == "FusedStagedReduction"
            )
        )

    def test_translated_lifetime_with_peak_reordering_and_memory_pools(self):
        inputs = _make_mla_inputs(batch_size=2, seq_len=8)
        eager = tuple(shifted_mla_indexer(*inputs))
        for reorder, memory_planning, memory_pool in (
            (False, False, "none"),
            (True, False, "none"),
            (False, True, "intermediates"),
            (True, True, "intermediates"),
        ):
            with self.subTest(
                reorder=reorder,
                memory_planning=memory_planning,
                memory_pool=memory_pool,
            ):
                observation = _observe(
                    shifted_mla_indexer,
                    inputs,
                    polyhedral_fusion=True,
                    reorder_for_peak_memory=reorder,
                    memory_planning=memory_planning,
                    memory_pool=memory_pool,
                )
                self.assertEqual(
                    observation.outputs,
                    eager,
                    atol=6e-2,
                    rtol=2e-2,
                )
                self.assert_translated_plan(observation)
                staged_lifetimes = [
                    signature
                    for signature in observation.lifetime_signatures
                    if signature[0] == "FusedStagedReduction"
                ]
                self.assertEqual(len(staged_lifetimes), 1)
                _, inner_nodes, outputs, last_usage = staged_lifetimes[0]
                self.assertGreater(inner_nodes, 1)
                self.assertGreater(outputs, 0)
                self.assertGreater(last_usage, 0)

    def test_translated_fallback_rejects_non_affine_layouts(self):
        inputs = _make_mla_inputs(batch_size=2, seq_len=8)
        for fn in (strided_shifted_mla_indexer, indirect_shifted_mla_indexer):
            with self.subTest(fn=fn.__name__):
                eager = tuple(fn(*inputs))
                observation = _observe(
                    fn,
                    inputs,
                    polyhedral_fusion=True,
                )
                self.assertEqual(
                    observation.outputs,
                    eager,
                    atol=6e-2,
                    rtol=2e-2,
                )
                self.assertEqual(observation.staged_fusion_count, 0)
                self.assertEqual(observation.translations, ())

    def test_legal_but_unsupported_split_declines(self):
        inputs = _make_mla_inputs(batch_size=2, seq_len=8)[:3]
        eager = tuple(equal_split_mla_indexer(*inputs))
        disabled = _observe(
            equal_split_mla_indexer,
            inputs,
            polyhedral_fusion=False,
        )
        enabled = _observe(
            equal_split_mla_indexer,
            inputs,
            polyhedral_fusion=True,
        )
        self.assertEqual(
            disabled.outputs, eager, atol=6e-2, rtol=2e-2
        )
        self.assertEqual(
            enabled.outputs, eager, atol=6e-2, rtol=2e-2
        )
        self.assertEqual(disabled.staged_fusion_count, 0)
        self.assertEqual(enabled.staged_fusion_count, 0)
        self.assertEqual(enabled.translations, ())


instantiate_parametrized_tests(PolyhedralMLAFusionTest)


if __name__ == "__main__":
    run_tests()
