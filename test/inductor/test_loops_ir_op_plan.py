# Owner(s): ["module: inductor"]

import dataclasses
import unittest
from types import SimpleNamespace as NS

import torch

from torch._inductor.codegen.loops_ir_op_options import (
    ArgRole,
    GeneratedKernel,
    KernelArg,
    LaunchSchema,
    NativeCall,
)
from torch._inductor.codegen.loops_ir_op_plan import (
    freeze_option_step,
    replace_option_step,
)
from torch._inductor.codegen.loops_ir_wrapper import (
    emit_loops_ir_wrapper,
    load_loops_ir_wrapper,
)
from torch._inductor.loops_ir_backend import (
    _ExternInput,
    _ExternResult,
    _ExternStep,
    _FrozenPlan,
    _KernelStep,
    _OutputPlan,
    _TensorKey,
)


def variant():
    return NS(
        input_bindings=[
            NS(root="a", root_offset=1, root_elements=8),
            NS(root="b", root_offset=0, root_elements=6),
        ],
        extra_bindings=[NS(root="extra", root_offset=3, root_elements=8)],
        output_binding=NS(root="y", root_offset=0, root_elements=4),
        signature=NS(output=NS(layout=NS(sizes=[2, 2], strides=[2, 1]))),
        fused_loops=NS(epilogue=None),
    )


def input_args():
    return (
        KernelArg(ArgRole.INPUT, "lhs", 0, (2, 3), (4, 1)),
        KernelArg(ArgRole.INPUT, "rhs", 1, (3, 2), (2, 1)),
    )


def generated(args=None, grid=(2, 3, 1)):
    if args is None:
        args = (
            *input_args(),
            KernelArg(ArgRole.OUTPUT, "out", 0, (2, 2), (2, 1), required_elements=4),
        )
    return GeneratedKernel(
        "test_generated",
        1,
        "key",
        "config",
        "selected_kernel",
        "source",
        LaunchSchema(grid, 4, 2, False, True, tuple(args)),
    )


def native(args=None):
    return NativeCall(
        "test_native",
        1,
        "key",
        "aten.mm.default",
        "selected_native",
        "def selected_native(lhs, rhs):\n    return torch.ops.aten.mm.default(lhs, rhs)\n",
        input_args() if args is None else tuple(args),
    )


class FreezeOptionStepTests(unittest.TestCase):
    def test_generated_bindings_preserve_view_offset_and_grid(self):
        frozen = freeze_option_step(generated(), variant())
        self.assertEqual(frozen.kind, "kernel")
        self.assertEqual(frozen.step.param_bases, ("a", "b", "y"))
        self.assertEqual(frozen.step.arg_views[0].rel, 1)
        self.assertEqual(frozen.step.arg_views[0].strides, (4, 1))
        self.assertEqual(frozen.step.grid, (2, 3, 1))
        self.assertFalse(frozen.step.enable_fp_fusion)
        self.assertFalse(frozen.step.enable_reflect_ftz)
        self.assertEqual(frozen.generated_span, ("y", 4))
        self.assertEqual(frozen.kernel_source, ("selected_kernel", "source"))

    def test_fused_extra_uses_root_pointer_not_double_offset(self):
        artifact = generated(
            args=(
                KernelArg(ArgRole.EXTRA_INPUT, "extra", 0, (8,), (1,)),
                KernelArg(ArgRole.OUTPUT, "out", 0, (2, 2), (2, 1)),
            )
        )
        frozen = freeze_option_step(artifact, variant())
        self.assertEqual(frozen.step.arg_views[0].base, "extra")
        self.assertEqual(frozen.step.arg_views[0].rel, 0)

    def test_shifted_generated_store_keeps_root_pointer(self):
        live = variant()
        live.output_binding.root_offset = 2
        live.output_binding.root_elements = 8
        artifact = generated(
            args=(
                *input_args(),
                KernelArg(
                    ArgRole.OUTPUT,
                    "out",
                    0,
                    (2, 2),
                    (2, 1),
                    required_elements=6,
                ),
            )
        )
        frozen = freeze_option_step(artifact, live)
        self.assertEqual(frozen.step.arg_views[-1].rel, 0)
        self.assertEqual(frozen.generated_span, ("y", 8))

    def test_shifted_native_root_is_rejected(self):
        live = variant()
        live.output_binding.root_offset = 1
        with self.assertRaisesRegex(ValueError, "offset-zero"):
            freeze_option_step(native(), live)

    def test_native_returns_output_without_preallocated_copy(self):
        frozen = freeze_option_step(native(), variant())
        self.assertEqual(frozen.kind, "extern")
        self.assertTrue(frozen.step.target_is_local)
        self.assertIsNone(frozen.step.target_fn)
        self.assertEqual(frozen.step.result_container, "tensor")
        self.assertEqual(frozen.step.results[0].name, "y")
        self.assertEqual(frozen.step.results[0].sizes, (2, 2))
        self.assertIsNone(frozen.generated_span)
        self.assertEqual(frozen.native_source[0], "selected_native")

    def test_frozen_bindings_do_not_retain_live_variant(self):
        live = variant()
        frozen = freeze_option_step(native(), live)
        live.input_bindings[0].root_offset = 999
        live.signature.output.layout.sizes[0] = 999
        self.assertEqual(frozen.step.inputs[0].rel, 1)
        self.assertEqual(frozen.step.results[0].sizes, (2, 2))

    def test_missing_prologue_operand_binding_rejects_rendered_read(self):
        live = variant()
        live.input_bindings[0] = None
        with self.assertRaisesRegex(ValueError, "no physical binding"):
            freeze_option_step(generated(), live)

    def test_input_extent_beyond_root_is_rejected(self):
        live = variant()
        live.input_bindings[0].root_elements = 7
        with self.assertRaisesRegex(ValueError, "exceeds"):
            freeze_option_step(generated(), live)

    def test_required_store_span_is_checked(self):
        artifact = generated(
            args=(
                *input_args(),
                KernelArg(
                    ArgRole.OUTPUT,
                    "out",
                    0,
                    (2, 2),
                    (2, 1),
                    required_elements=5,
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            freeze_option_step(artifact, variant())

    def test_shifted_native_output_is_rejected(self):
        live = variant()
        live.fused_loops.epilogue = NS(store_offset=1)
        with self.assertRaisesRegex(ValueError, "shifted"):
            freeze_option_step(native(), live)

    def test_empty_grid_skips_launch(self):
        frozen = freeze_option_step(generated(grid=(0, 1, 1)), variant())
        self.assertTrue(frozen.step.skip_empty)

    def test_negative_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "grid"):
            freeze_option_step(generated(grid=(-1, 1, 1)), variant())

    def test_generated_output_must_be_last(self):
        artifact = generated(
            args=(
                KernelArg(ArgRole.OUTPUT, "out", 0, (2, 2), (2, 1)),
                *input_args(),
            )
        )
        with self.assertRaisesRegex(ValueError, "final output"):
            freeze_option_step(artifact, variant())

    def test_native_output_pointer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "return its output"):
            freeze_option_step(
                native(
                    args=(
                        KernelArg(
                            ArgRole.OUTPUT,
                            "out",
                            0,
                            (2, 2),
                            (2, 1),
                        ),
                    )
                ),
                variant(),
            )

    def test_negative_operand_slot_is_rejected(self):
        bad = dataclasses.replace(input_args()[0], slot=-1)
        with self.assertRaisesRegex(ValueError, "slot"):
            freeze_option_step(native(args=(bad, input_args()[1])), variant())

    def test_native_output_extent_is_checked(self):
        live = variant()
        live.output_binding.root_elements = 3
        with self.assertRaisesRegex(ValueError, "output extent"):
            freeze_option_step(native(), live)


def baseline_plan():
    mm = _ExternStep(
        "mm",
        None,
        "aten.mm.default",
        (
            _ExternInput("lhs", "a", 1, (2, 3), (4, 1), 8),
            _ExternInput("rhs", "b", 0, (3, 2), (2, 1), 6),
        ),
        (_TensorKey("lhs"), _TensorKey("rhs")),
        (),
        (_ExternResult("mm", torch.float32, (2, 2), (2, 1)),),
        "tensor",
    )
    relu = _KernelStep(
        "baseline_relu", ("mm", "y"), (1, 1, 1), 4, False, False, False, None
    )
    return _FrozenPlan(
        device_index=0,
        num_inputs=2,
        input_bases=("a", "b"),
        gen_spans=(("y", 4),),
        span_full_coverage=(True,),
        steps=[("extern", mm), ("kernel", relu)],
        out_plans=(_OutputPlan("y", 0, (2, 2), (2, 1)),),
        out_spec=None,
        kernel_sources={"baseline_relu": "baseline source"},
        kernel_names=["baseline_relu"],
        kernel_params={"baseline_relu": ["mm", "y"]},
        kernel_grids={"baseline_relu": (1, 1, 1)},
        input_layouts=(("a", (8,), (1,)), ("b", (3, 2), (2, 1))),
        result_layouts=(("mm", (2, 2), (2, 1)),),
        num_generated_kernels=1,
        num_extern_calls=1,
        kernels_before_fusion=1,
        program_dump="",
        optimized_program_dump="",
        kernel_dump="",
        tiled_dump="",
    )


class ReplaceOptionStepTests(unittest.TestCase):
    def apply(self, plan, option=None, **overrides):
        arguments = dict(
            consumed_buffers=frozenset(("mm", "y")),
            step_outputs=(("mm",), ("y",)),
            step_positions=(2, 3),
            placement=3,
        )
        arguments.update(overrides)
        if option is None:
            live = variant()
            live.fused_loops.epilogue = NS(store_offset=0)
            artifact = dataclasses.replace(
                native(),
                source=(
                    "def selected_native(lhs, rhs):\n"
                    "    return torch.relu(torch.mm(lhs, rhs))\n"
                ),
            )
            option = freeze_option_step(artifact, live)
        return replace_option_step(plan, option, **arguments)

    def test_whole_region_replaces_steps_without_copy_or_stale_sources(self):
        baseline = baseline_plan()
        updated, outputs, positions = self.apply(baseline)
        self.assertEqual(len(baseline.steps), 2)
        self.assertEqual(len(updated.steps), 1)
        self.assertEqual(updated.steps[0][0], "extern")
        self.assertEqual(updated.gen_spans, ())
        self.assertEqual(updated.kernel_sources, {})
        self.assertEqual(updated.kernel_names, [])
        self.assertEqual(updated.num_generated_kernels, 0)
        self.assertEqual(updated.num_extern_calls, 1)
        self.assertEqual(updated.result_layouts, (("y", (2, 2), (2, 1)),))
        self.assertEqual(outputs, (("y",),))
        self.assertEqual(positions, (3,))

    def test_generated_output_gets_span_but_not_stale_native_layout(self):
        updated, _, _ = self.apply(
            baseline_plan(), freeze_option_step(generated(), variant())
        )
        self.assertEqual(updated.gen_spans, (("y", 4),))
        self.assertEqual(updated.span_full_coverage, (False,))
        self.assertEqual(updated.result_layouts, ())
        self.assertEqual(updated.kernel_names, ["selected_kernel"])
        self.assertEqual(updated.num_generated_kernels, 1)
        self.assertEqual(updated.num_extern_calls, 0)

    def test_partial_multioutput_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "partially consumes"):
            self.apply(baseline_plan(), step_outputs=(("mm",), ("y", "other")))

    def test_graph_output_cannot_be_eliminated(self):
        baseline = baseline_plan()._replace(
            out_plans=(_OutputPlan("mm", 0, (2, 2), (2, 1)),)
        )
        with self.assertRaisesRegex(ValueError, "graph output"):
            self.apply(baseline)

    def test_absorbed_prologue_frontier_cannot_be_read(self):
        live = variant()
        live.input_bindings[0].root = "absorbed_temp"
        option = freeze_option_step(native(), live)
        with self.assertRaisesRegex(ValueError, "frontier is unavailable"):
            self.apply(baseline_plan(), option)

    def test_unclaimed_group_cannot_keep_reading_removed_anchor(self):
        baseline = baseline_plan()
        extra = _KernelStep(
            "tail", ("mm", "z"), (1, 1, 1), 4, False, False, False, None
        )
        baseline = baseline._replace(steps=[*baseline.steps, ("kernel", extra)])
        with self.assertRaisesRegex(ValueError, "remaining step reads"):
            self.apply(
                baseline,
                step_outputs=(("mm",), ("y",), ("z",)),
                step_positions=(2, 3, 4),
            )

    def test_late_placement_cannot_delay_visible_consumer(self):
        baseline = baseline_plan()
        extra = _KernelStep("tail", ("y", "z"), (1, 1, 1), 4, False, False, False, None)
        baseline = baseline._replace(steps=[*baseline.steps, ("kernel", extra)])
        with self.assertRaisesRegex(ValueError, "needed before"):
            self.apply(
                baseline,
                step_outputs=(("mm",), ("y",), ("z",)),
                step_positions=(2, 3, 4),
                placement=5,
            )

    def test_graph_input_cannot_be_claimed_as_an_intermediate(self):
        with self.assertRaisesRegex(ValueError, "graph input"):
            self.apply(baseline_plan(), consumed_buffers=frozenset(("a", "mm", "y")))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FreezeOptionExecutionTests(unittest.TestCase):
    def test_replaced_mm_relu_region_executes_without_original_kernel(self):
        from torch._inductor.loops_ir_backend import _emit_and_load_wrapper

        live = variant()
        live.fused_loops.epilogue = NS(store_offset=0)
        artifact = dataclasses.replace(
            native(),
            source=(
                "def selected_native(lhs, rhs):\n"
                "    return torch.relu(torch.mm(lhs, rhs))\n"
            ),
        )
        option = freeze_option_step(artifact, live)
        updated, _, _ = replace_option_step(
            baseline_plan(),
            option,
            consumed_buffers=frozenset(("mm", "y")),
            step_outputs=(("mm",), ("y",)),
            step_positions=(2, 3),
            placement=3,
        )
        self.assertEqual(updated.kernel_sources, {})
        module, _ = _emit_and_load_wrapper(updated)
        a = torch.arange(8, dtype=torch.float32, device="cuda") - 4
        b = torch.arange(6, dtype=torch.float32, device="cuda").reshape(3, 2) - 2
        expected = torch.relu(torch.mm(torch.as_strided(a, (2, 3), (4, 1), 1), b))
        actual = module.call([a, b])[0]
        self.assertTrue(
            torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        )

    def test_frozen_native_mm_executes_through_production_wrapper(self):
        frozen = freeze_option_step(native(), variant())
        source = emit_loops_ir_wrapper(
            kernel_sources={},
            kernel_order=[],
            kernel_params={},
            kernel_grids={},
            steps=[(frozen.kind, frozen.step)],
            input_bases=("a", "b"),
            gen_spans=(),
            out_plans=(_OutputPlan("y", 0, (2, 2), (2, 1)),),
            num_inputs=2,
            device_index=torch.cuda.current_device(),
            native_sources=(frozen.native_source,),
            result_layouts=(frozen.result_layout,),
            input_layouts=(("a", (8,), (1,)), ("b", (3, 2), (2, 1))),
        )
        mod, _, _, _ = load_loops_ir_wrapper(source)
        a = torch.arange(8, dtype=torch.float32, device="cuda")
        b = torch.arange(6, dtype=torch.float32, device="cuda").reshape(3, 2)
        expected = torch.mm(torch.as_strided(a, (2, 3), (4, 1), 1), b)
        actual = mod.call([a, b])[0]
        self.assertTrue(
            torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        )
        self.assertEqual(actual.stride(), expected.stride())


if __name__ == "__main__":
    unittest.main()
