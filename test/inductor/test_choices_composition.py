# Owner(s): ["module: inductor"]

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from torch._inductor import config, ir, select_algorithm
from torch._inductor.choices import (
    create_inductor_choices,
    InductorChoices,
    MultiTemplateFusionDecision,
    register_inductor_choices,
    registered_inductor_choices,
    unregister_inductor_choices,
)
from torch._inductor.scheduler import Scheduler
from torch.utils._ordered_set import OrderedSet
from torch._inductor.virtualized import _choices, threadlocal, V
from torch.testing._internal.common_utils import run_tests, TestCase


_UNSET = object()


class AlphaChoices(InductorChoices):
    suffix = "_alpha"

    def __init__(self, threshold: int = 7) -> None:
        super().__init__()
        self.result_suffix = f"{self.suffix}:{threshold}"

    def uuid(self) -> str:
        return f"alpha:{self.result_suffix}"

    def customize_fused_kernel_name(self, fused_name: str, src_code: str) -> str:
        return f"{fused_name}{self.result_suffix}"


class _BetaHeuristics:
    def get_conv_configs(self) -> str:
        return "beta"


class BetaChoices(InductorChoices):
    def __init__(self) -> None:
        super().__init__()
        self.heuristics = _BetaHeuristics()

    def uuid(self) -> str:
        return "beta"

    def get_config_heuristics(self, device_type: str | None = "cuda") -> Any:
        return self.heuristics


class SecondNameChoices(InductorChoices):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def uuid(self) -> str:
        return "second"

    def customize_fused_kernel_name(self, fused_name: str, src_code: str) -> str:
        self.calls += 1
        return f"{fused_name}_second"


class ForwardFlexChoices(InductorChoices):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def uuid(self) -> str:
        return "forward-flex"

    def append_flex_attention_choices(self, choices, *args, **kwargs):
        self.calls += 1
        return choices


class BackwardFlexChoices(InductorChoices):
    def __init__(self) -> None:
        super().__init__()
        self.mutated_inputs: list[Any] | None = None

    def uuid(self) -> str:
        return "backward-flex"

    def append_flex_attention_backward_choices(
        self, choices, *args, mutated_inputs: list[Any], **kwargs
    ):
        self.mutated_inputs = mutated_inputs
        return choices


class UUIDChoices(InductorChoices):
    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value

    def uuid(self) -> str:
        return self.value


class NotChoices:
    def uuid(self) -> str:
        return "not-choices"


class PlainChoices(InductorChoices):
    pass


class StaticFalseChoices(InductorChoices):
    def uuid(self) -> str:
        return "static-false"

    @staticmethod
    def can_fuse(
        scheduler: Any,
        node1: Any,
        node2: Any,
        shared_data_score: int,
    ) -> bool:
        return False


class StaticTrueChoices(InductorChoices):
    def uuid(self) -> str:
        return "static-true"

    @staticmethod
    def can_fuse(
        scheduler: Any,
        node1: Any,
        node2: Any,
        shared_data_score: int,
    ) -> bool:
        return True


class ChoicesCompositionTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._prior_factory = config.inductor_choices_class
        self._prior_handler = getattr(threadlocal, _choices._key, _UNSET)
        self._prior_registrations = registered_inductor_choices()
        config.inductor_choices_class = None
        for key, _ in self._prior_registrations:
            unregister_inductor_choices(key)
        if hasattr(threadlocal, _choices._key):
            delattr(threadlocal, _choices._key)

    def tearDown(self) -> None:
        config.inductor_choices_class = None
        for key, _ in registered_inductor_choices():
            unregister_inductor_choices(key)
        for key, factory in self._prior_registrations:
            register_inductor_choices(key, factory)
        config.inductor_choices_class = self._prior_factory
        if hasattr(threadlocal, _choices._key):
            delattr(threadlocal, _choices._key)
        if self._prior_handler is not _UNSET:
            setattr(threadlocal, _choices._key, self._prior_handler)
        super().tearDown()

    def current_choices(self) -> InductorChoices:
        return create_inductor_choices(config.inductor_choices_class)

    def test_no_contributors_returns_base(self) -> None:
        self.assertIs(type(self.current_choices()), InductorChoices)

    def test_existing_config_factory_is_returned_untouched(self) -> None:
        part = AlphaChoices(threshold=3)
        with config.patch(inductor_choices_class=lambda: part):
            self.assertIs(self.current_choices(), part)

    def test_direct_virtualized_handler_is_unchanged(self) -> None:
        direct = AlphaChoices(threshold=3)
        with V.set_choices_handler(direct):
            self.assertIs(V.choices, direct)

    def test_registered_choices_compose_in_order(self) -> None:
        register_inductor_choices("alpha", lambda: AlphaChoices(threshold=99))
        register_inductor_choices("beta", BetaChoices)

        handler = V.choices
        self.assertEqual(
            handler.customize_fused_kernel_name("kernel", ""), "kernel_alpha:99"
        )
        self.assertEqual(handler.get_conv_configs(), "beta")
        expected_uuid = (
            "composed_inductor_choices",
            ("alpha:_alpha:99", "beta"),
        )
        self.assertEqual(handler.uuid(), expected_uuid)
        self.assertEqual(
            config.save_config_portable(ignore_private_configs=False)[
                "inductor_choices_class"
            ],
            expected_uuid,
        )

    def test_config_factory_precedes_registered_choices(self) -> None:
        second = SecondNameChoices()
        register_inductor_choices("second", lambda: second)

        with config.patch(inductor_choices_class=AlphaChoices):
            handler = self.current_choices()
            with self.assertLogs("torch._inductor.choices", level="WARNING"):
                self.assertEqual(
                    handler.customize_fused_kernel_name("kernel", ""),
                    "kernel_alpha:7",
                )

        self.assertEqual(second.calls, 0)

    def test_conflicting_override_warns_and_first_wins(self) -> None:
        second = SecondNameChoices()
        register_inductor_choices("alpha", AlphaChoices)
        register_inductor_choices("second", lambda: second)
        handler = V.choices

        with self.assertLogs("torch._inductor.choices", level="WARNING") as logs:
            self.assertEqual(
                handler.customize_fused_kernel_name("kernel", ""),
                "kernel_alpha:7",
            )
        self.assertEqual(second.calls, 0)
        self.assertIn("customize_fused_kernel_name", "\n".join(logs.output))

    def test_flex_attention_forward_override_is_not_used_for_backward(self) -> None:
        handler = ForwardFlexChoices()
        forward_choices = [object()]
        backward_choices = [object()]
        args = ([], [], [], None, {}, 1, 1)

        self.assertIs(
            handler.append_flex_attention_choices(forward_choices, *args),
            forward_choices,
        )
        self.assertIs(
            handler.append_flex_attention_backward_choices(
                backward_choices,
                *args,
                mutated_inputs=[],
            ),
            backward_choices,
        )
        self.assertEqual(handler.calls, 1)

    def test_flex_attention_hooks_compose_independently(self) -> None:
        forward = ForwardFlexChoices()
        backward = BackwardFlexChoices()
        register_inductor_choices("forward", lambda: forward)
        register_inductor_choices("backward", lambda: backward)
        mutated_inputs = [object()]
        handler = V.choices

        forward_choices = [object()]
        backward_choices = [object()]
        args = ([], [], [], None, {}, 1, 1)
        self.assertIs(
            handler.append_flex_attention_choices(forward_choices, *args),
            forward_choices,
        )
        self.assertIs(
            handler.append_flex_attention_backward_choices(
                backward_choices,
                *args,
                mutated_inputs=mutated_inputs,
            ),
            backward_choices,
        )

        self.assertEqual(forward.calls, 1)
        self.assertIs(backward.mutated_inputs, mutated_inputs)

    def test_flex_attention_backward_mutated_inputs_is_keyword_only(self) -> None:
        choices = InductorChoices()
        with self.assertRaises(TypeError):
            choices.append_flex_attention_backward_choices(
                [], [], [], [], None, {}, 1, 1, []
            )

    def test_staticmethod_conflict_warns_once(self) -> None:
        register_inductor_choices("false", StaticFalseChoices)
        register_inductor_choices("true", StaticTrueChoices)
        handler = V.choices

        with self.assertLogs("torch._inductor.choices", level="WARNING") as logs:
            self.assertFalse(handler.can_fuse(None, None, None, 0))
            self.assertFalse(handler.can_fuse(None, None, None, 0))

        self.assertEqual(len(logs.output), 1)
        self.assertIn("StaticFalseChoices", logs.output[0])
        self.assertIn("StaticTrueChoices", logs.output[0])

    def test_dispatch_target_is_cached(self) -> None:
        register_inductor_choices("alpha", AlphaChoices)
        register_inductor_choices("beta", BetaChoices)
        handler = V.choices

        self.assertIs(
            handler.customize_fused_kernel_name,
            handler.customize_fused_kernel_name,
        )

    def test_runtime_instance_override_is_detected(self) -> None:
        choice = UUIDChoices("runtime")
        choice.__dict__["customize_fused_kernel_name"] = (
            lambda fused_name, src_code: f"{fused_name}_runtime"
        )

        register_inductor_choices("beta", BetaChoices)
        register_inductor_choices("runtime", lambda: choice)
        handler = V.choices

        self.assertEqual(
            handler.customize_fused_kernel_name("kernel", ""),
            "kernel_runtime",
        )

    def test_instance_patch_takes_precedence(self) -> None:
        register_inductor_choices("alpha", AlphaChoices)
        register_inductor_choices("beta", BetaChoices)
        handler = V.choices

        with patch.object(
            handler,
            "customize_fused_kernel_name",
            lambda fused_name, src_code: f"{fused_name}_patched",
        ):
            self.assertEqual(
                handler.customize_fused_kernel_name("kernel", ""),
                "kernel_patched",
            )

        self.assertEqual(
            handler.customize_fused_kernel_name("kernel", ""),
            "kernel_alpha:7",
        )

    def test_fresh_thread_uses_registry_and_matching_cache_key(self) -> None:
        register_inductor_choices("alpha", AlphaChoices)
        register_inductor_choices("beta", BetaChoices)

        def get_runtime_and_cache_uuid() -> tuple[Any, Any]:
            return (
                V.choices.uuid(),
                config.save_config_portable(ignore_private_configs=False)[
                    "inductor_choices_class"
                ],
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            runtime_uuid, cache_uuid = executor.submit(
                get_runtime_and_cache_uuid
            ).result()

        self.assertEqual(runtime_uuid, cache_uuid)
        self.assertEqual(
            runtime_uuid,
            ("composed_inductor_choices", ("alpha:_alpha:7", "beta")),
        )

    def test_inherited_default_is_used_when_nobody_overrides(self) -> None:
        register_inductor_choices("beta", BetaChoices)
        register_inductor_choices("uuid", lambda: UUIDChoices("uuid-only"))
        self.assertEqual(
            V.choices.customize_fused_kernel_name("kernel", ""),
            "kernel",
        )

    def test_composed_uuid_is_structured_and_ordered(self) -> None:
        register_inductor_choices("first", lambda: UUIDChoices("a+b"))
        register_inductor_choices("second", lambda: UUIDChoices("c"))
        forward = V.choices.uuid()
        unregister_inductor_choices("first")
        unregister_inductor_choices("second")
        register_inductor_choices("second", lambda: UUIDChoices("c"))
        register_inductor_choices("first", lambda: UUIDChoices("a+b"))
        reverse = V.choices.uuid()

        self.assertEqual(
            forward,
            (
                "composed_inductor_choices",
                ("a+b", "c"),
            ),
        )
        self.assertNotEqual(forward, reverse)

    def test_composed_uuid_has_no_delimiter_collisions(self) -> None:
        register_inductor_choices("first", lambda: UUIDChoices("a+b"))
        register_inductor_choices("second", lambda: UUIDChoices("c"))
        ab_c = V.choices.uuid()
        unregister_inductor_choices("first")
        unregister_inductor_choices("second")
        register_inductor_choices("first", lambda: UUIDChoices("a"))
        register_inductor_choices("second", lambda: UUIDChoices("b+c"))
        a_bc = V.choices.uuid()
        self.assertNotEqual(ab_c, a_bc)

    def test_composed_uuid_tracks_contributor_state(self) -> None:
        register_inductor_choices("beta", BetaChoices)
        register_inductor_choices("alpha", lambda: AlphaChoices(threshold=1))
        first = V.choices.uuid()
        register_inductor_choices("alpha", lambda: AlphaChoices(threshold=2))
        second = V.choices.uuid()
        self.assertNotEqual(first, second)

    def test_non_choices_contributor_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "must return InductorChoices"):
            register_inductor_choices("invalid", lambda: _as_choices(NotChoices()))
        self.assertEqual(registered_inductor_choices(), ())

    def test_missing_uuid_is_rejected_at_construction(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not implement uuid"):
            register_inductor_choices("plain", PlainChoices)
        self.assertEqual(registered_inductor_choices(), ())

    def test_registration_replaces_in_place(self) -> None:
        register_inductor_choices("alpha", lambda: AlphaChoices(threshold=1))
        register_inductor_choices("beta", BetaChoices)
        register_inductor_choices("alpha", lambda: AlphaChoices(threshold=2))

        self.assertEqual(
            tuple(key for key, _ in registered_inductor_choices()),
            ("alpha", "beta"),
        )
        self.assertEqual(
            V.choices.uuid(),
            ("composed_inductor_choices", ("alpha:_alpha:2", "beta")),
        )


def _as_choices(obj: Any) -> InductorChoices:
    return obj


class RecordingFusionChoices(InductorChoices):
    """Returns a fixed MultiTemplateFusionDecision and records the call."""

    def __init__(self, decision: MultiTemplateFusionDecision | None) -> None:
        super().__init__()
        self.decision = decision
        self.calls: list[tuple[Any, list[Any], bool]] = []

    def uuid(self) -> str:
        return "recording-fusion"

    def multi_template_fusion_decision(
        self, scheduler: Any, multi_node: Any, other_nodes: Any, epilogue: bool
    ) -> MultiTemplateFusionDecision | None:
        self.calls.append((multi_node, list(other_nodes), epilogue))
        return self.decision


class MultiTemplateFusionDecisionTest(TestCase):
    """V.choices.multi_template_fusion_decision through the real fusion loop:
    Scheduler._try_fusion_pairs -> speedup_by_fusion -> the hook, with a real
    MultiTemplateBuffer state and finalization. Legality inputs (can_fuse, cycle
    and memory checks) are stubbed; fuse_two_nodes records commits."""

    def setUp(self) -> None:
        super().setUp()
        self.constraints: dict[str, Any] = {}
        names = iter(f"buf{i}" for i in range(100))
        graph = SimpleNamespace(
            buffer_layout_constraints=self.constraints,
            register_buffer=lambda buffer, set_name=False: next(names),
            register_operation=lambda op: "op0",
        )
        for ctx in (
            V.set_graph_handler(graph),
            patch(
                "torch._inductor.scheduler._is_atomic_add_mutation_epilogue",
                return_value=False,
            ),
            config.patch(multi_kernel_hints=[]),
        ):
            ctx.__enter__()
            self.addCleanup(ctx.__exit__, None, None, None)

        self.layout = ir.FixedLayout(torch.device("cuda"), torch.float16, [64, 32], [32, 1])
        self.t1, self.t2 = self._template_caller(), self._template_caller()
        self.extern = MagicMock(spec=select_algorithm.ExternKernelCaller)
        self.extern.has_out_variant = True
        self.multi = ir.MultiTemplateBuffer(
            self.layout,
            [],
            lambda hint: {},
            [self.extern, self.t1, self.t2],
            OrderedSet(),
            OrderedSet(),
        )
        self.template = self._snode("template", template=self.multi)
        self.epilogues = [self._snode(f"epi{i}") for i in range(3)]
        self.commits: list[tuple[Any, Any, Any]] = []

        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.seen_template_fusions = OrderedSet()
        self.scheduler._fusion_memory_state = None
        for name, value in (
            ("get_fused_node", lambda n: n),
            ("can_fuse", lambda n1, n2, can_reorder=False: True),
            ("will_fusion_create_cycle", lambda n1, n2: False),
            ("fuse_two_nodes", self._record_commit),
            ("_any_atomic_add", lambda nodes: False),
        ):
            setattr(self.scheduler, name, value)

    def _template_caller(self) -> Any:
        caller = MagicMock(spec=select_algorithm.TritonTemplateCaller)
        caller.layout = self.layout
        caller.load_input_fusion_allowed_inputs = OrderedSet()
        caller.store_output_fusion_allowed_inputs = OrderedSet()
        caller.get_make_kernel_render.return_value = object()
        return caller

    def _snode(self, name: str, template: Any = None) -> Any:
        node = MagicMock()
        node.get_name.return_value = name
        node.get_first_name.return_value = name
        node.is_template.return_value = template is not None
        node.get_template_node.return_value = template
        node.get_nodes.return_value = [node]
        node.get_device.return_value = torch.device("cuda")
        node.is_foreach.return_value = False
        return node

    def _record_commit(self, node1: Any, node2: Any, fused_nodes: Any) -> Any:
        self.commits.append((node1, node2, self.multi._render_caller))
        return node1

    def _run(self, choices: InductorChoices, pairs: list[tuple[Any, Any]]) -> None:
        with V.set_choices_handler(choices):
            self.scheduler._try_fusion_pairs(pairs, {}, {}, OrderedSet(), False)

    def _decide(self, decision: Any, prologue: bool = False) -> RecordingFusionChoices:
        choices = RecordingFusionChoices(decision)
        epi = self.epilogues[0]
        self._run(choices, [(epi, self.template) if prologue else (self.template, epi)])
        return choices

    def assert_not_committed(self, render_caller: Any = None) -> None:
        self.assertEqual(self.commits, [])
        self.assertIs(self.multi._render_caller, render_caller)

    def test_accept_commits_and_finalizes_chosen_template(self) -> None:
        choices = self._decide(MultiTemplateFusionDecision(fuse=True, choice=self.t2))
        self.assertEqual(self.commits, [(self.template, self.epilogues[0], self.t2)])
        self.assertIs(self.multi.make_kernel_render, self.t2.get_make_kernel_render.return_value)
        self.assertEqual(choices.calls, [(self.multi, [self.epilogues[0]], True)])

    def test_decline_does_not_commit_or_finalize(self) -> None:
        self._decide(MultiTemplateFusionDecision(fuse=False))
        self.assert_not_committed()
        self.assertIsNone(self.multi.make_kernel_render)

    def test_layout_conflict_blocks_accepted_fusion(self) -> None:
        inp = MagicMock()
        inp.get_name.return_value = "arg0"
        inp.layout = ir.FixedLayout(torch.device("cuda"), torch.float16, [64, 32], [1, 64])
        self.multi.inputs = [inp]
        self.constraints["arg0"] = ir.FixedLayout(
            torch.device("cuda"), torch.float16, [64, 32], [32, 1]
        )
        with patch.object(select_algorithm, "should_use_layout_constraints", return_value=True):
            self._decide(MultiTemplateFusionDecision(fuse=True, choice=self.t1))
        self.assert_not_committed()

    def test_accept_without_choice_on_unfinalized_buffer_is_declined(self) -> None:
        self._decide(MultiTemplateFusionDecision(fuse=True))
        self.assert_not_committed()

    def test_accept_without_choice_reuses_finalized_template(self) -> None:
        self.multi.finalize_as_triton_caller(self.t1)
        self._decide(MultiTemplateFusionDecision(fuse=True))
        self.assertEqual(self.commits, [(self.template, self.epilogues[0], self.t1)])

    def test_extern_choice_is_declined(self) -> None:
        self._decide(MultiTemplateFusionDecision(fuse=True, choice=self.extern))
        self.assert_not_committed()

    def test_caller_of_another_buffer_is_declined(self) -> None:
        self._decide(MultiTemplateFusionDecision(fuse=True, choice=self._template_caller()))
        self.assert_not_committed()

    def test_choice_with_other_layout_is_declined(self) -> None:
        self.t2.layout = ir.FixedLayout(torch.device("cuda"), torch.float16, [64, 32], [1, 64])
        self._decide(MultiTemplateFusionDecision(fuse=True, choice=self.t2))
        self.assert_not_committed()

    def test_conflicting_finalized_choice_is_declined(self) -> None:
        self.multi.finalize_as_triton_caller(self.t1)
        self._decide(MultiTemplateFusionDecision(fuse=True, choice=self.t2))
        self.assert_not_committed(render_caller=self.t1)

    def test_atomic_add_prologue_is_declined(self) -> None:
        self.scheduler._any_atomic_add = lambda nodes: True
        self._decide(MultiTemplateFusionDecision(fuse=True, choice=self.t1), prologue=True)
        self.assert_not_committed()

    def test_unsupported_prologue_input_is_declined(self) -> None:
        self.multi.allow_prologue_fusion = True
        self.multi.load_input_fusion_allowed_inputs = OrderedSet(["arg0"])
        choices = self._decide(MultiTemplateFusionDecision(fuse=True, choice=self.t1), prologue=True)
        self.assert_not_committed()
        self.assertEqual(choices.calls, [(self.multi, [self.epilogues[0]], False)])

    def test_unsupported_store_output_input_is_declined(self) -> None:
        # A producer of an input the template fuses in store_output (e.g. the
        # addmm bias) that the chosen template does not support.
        self.multi.allow_epilogue_fusion = True
        self.multi.store_output_fusion_allowed_inputs = OrderedSet(["bias"])
        self._decide(MultiTemplateFusionDecision(fuse=True, choice=self.t1), prologue=True)
        self.assert_not_committed()

    def test_commit_order_in_one_round(self) -> None:
        """First pair declined, second committed with t1, third asks for t2 after
        the buffer was finalized and is declined."""
        decisions = iter([
            MultiTemplateFusionDecision(fuse=False),
            MultiTemplateFusionDecision(fuse=True, choice=self.t1),
            MultiTemplateFusionDecision(fuse=True, choice=self.t2),
        ])

        class Sequenced(RecordingFusionChoices):
            def multi_template_fusion_decision(self, *args: Any) -> Any:
                super().multi_template_fusion_decision(*args)
                return next(decisions)

        choices = Sequenced(None)
        self._run(choices, [(self.template, e) for e in self.epilogues])
        self.assertEqual(self.commits, [(self.template, self.epilogues[1], self.t1)])
        self.assertEqual(len(choices.calls), 3)
        self.assertIs(self.multi._render_caller, self.t1)

    def _reach_upstream_benchmark_path(self, choices: InductorChoices) -> Any:
        """Stops the benchmark-based path at its layout-conflict check."""
        self.scheduler.get_backend = lambda device: SimpleNamespace(
            has_sub_parent_epilogue=lambda nodes: False
        )
        upstream = MagicMock(return_value=True)
        self.scheduler._has_layout_conflict_for_template = upstream
        self._run(choices, [(self.template, self.epilogues[0])])
        return upstream

    def test_default_choices_keep_the_benchmark_path(self) -> None:
        upstream = self._reach_upstream_benchmark_path(PlainChoices())
        upstream.assert_called_once_with(self.multi)
        self.assert_not_committed()

    def test_hook_not_consulted_with_multi_kernel_hints(self) -> None:
        choices = RecordingFusionChoices(MultiTemplateFusionDecision(fuse=True, choice=self.t1))
        with config.patch(multi_kernel_hints=[64]):
            upstream = self._reach_upstream_benchmark_path(choices)
        self.assertEqual(choices.calls, [])
        upstream.assert_called_once_with(self.multi)
        self.assert_not_committed()


if __name__ == "__main__":
    run_tests()
