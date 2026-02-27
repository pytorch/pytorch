# Owner(s): ["module: dynamo"]
import dataclasses
import os
import pprint
import sys
from unittest import mock

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch.compiler.config as compiler_config
from torch._dynamo import utils
from torch._inductor.test_case import TestCase


_IS_WINDOWS = sys.platform == "win32"


class TestUtils(TestCase):
    def test_nan(self):
        a = torch.Tensor([float("nan")])
        b = torch.Tensor([float("nan")])
        fp64_ref = torch.DoubleTensor([5.0])
        res = utils.same(a, b, fp64_ref=fp64_ref, equal_nan=True)
        self.assertTrue(res)

    def test_larger_multiplier_for_smaller_tensor(self):
        """
        Tensor numel between (10, 500]
        """
        N = 100
        fp64_ref = torch.full([N], 0.0, dtype=torch.double)
        a = torch.full([N], 1.0)
        tol = 4 * 1e-2
        self.assertTrue(utils.same(a, a * 2, fp64_ref=fp64_ref, tol=tol))
        self.assertFalse(utils.same(a, a * 4, fp64_ref=fp64_ref, tol=tol))
        self.assertTrue(
            utils.same(
                a,
                a * 4,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )
        self.assertFalse(
            utils.same(
                a,
                a * 9,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )

    def test_larger_multiplier_for_even_smaller_tensor(self):
        """
        Tensor numel <=10
        """
        fp64_ref = torch.DoubleTensor([0.0])
        a = torch.Tensor([1.0])
        tol = 4 * 1e-2
        self.assertTrue(utils.same(a, a * 2, fp64_ref=fp64_ref, tol=tol))
        self.assertFalse(utils.same(a, a * 7, fp64_ref=fp64_ref, tol=tol))
        self.assertTrue(
            utils.same(
                a,
                a * 7,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )
        self.assertFalse(
            utils.same(
                a,
                a * 20,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )

    @dynamo_config.patch(
        {
            "log_compilation_metrics": True,
            "inline_inbuilt_nn_modules": False,
        }
    )
    def test_graph_break_counting(self):
        """
        Run a compilation that includes a graph break and validate that the
        graph break counter is incremented.
        """

        def run_forward_backward():
            model = torch.compile(TestModel())
            x = torch.rand([3], requires_grad=True)
            output = model(x)
            loss_fn = torch.nn.MSELoss()
            target = torch.tensor([1.0])
            loss = loss_fn(output, target)
            loss.backward()

        @torch.compile(backend="eager")
        def add(x, y):
            return x + y

        @torch.compile(backend="eager")
        def break_it(x):
            y = x.sum()
            if y > 0:
                return x + y.item()
            return x - y.item()

        @torch.compile(backend="eager")
        def break_it2(x):
            y = x.sum()
            if y > 0:
                if y > 1:
                    return x * y.item()
                return x + y.item()
            return x - y.item()

        add(torch.rand([10]), torch.rand([10]))
        utils.reset_frame_count()

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            run_forward_backward()
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
            self.assertEqual(compilation_events[-1].num_graph_breaks, 0)

            # We should fallback to normal mode and increment the graph break counter
            torch.compile(break_it, backend="inductor")(torch.ones(3, 3))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
            self.assertEqual(compilation_events[-1].num_graph_breaks, 1)

            # Graph break counter should be incremented by 1 (after a reset), not 2
            torch.compile(break_it, backend="inductor")(torch.ones(3, 3))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
            self.assertEqual(compilation_events[-1].num_graph_breaks, 1)

            # Graph break counter should be incremented by 2
            torch.compile(break_it2, backend="inductor")(torch.ones(3, 3))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
            self.assertEqual(compilation_events[-1].num_graph_breaks, 2)

    def test_traced_code_query(self):
        try:
            from .utils import add, break_it
        except ImportError:
            from utils import add, break_it

        traced_code_lists = []

        def get_filenames(traced_code_lists):
            return [
                [code.co_filename for code in code_list]
                for code_list in traced_code_lists
            ]

        def my_backend(gm, example_inputs):
            from torch._dynamo.utils import get_traced_code

            nonlocal traced_code_lists
            traced_code_lists.append(get_traced_code())
            return gm.forward

        utils_path = os.path.join(os.path.dirname(__file__), "utils.py")

        # === no inlining ===
        @torch.compile(backend=my_backend)
        def fn(x):
            return x * 2

        x = torch.randn(3)
        traced_code_lists = []
        fn(x)
        self.assertEqual(get_filenames(traced_code_lists), [[__file__]])

        # === successful inlining ===
        @torch.compile(backend=my_backend)
        def fn(x):
            return add(x) * 2

        x = torch.randn(3)
        traced_code_lists = []
        fn(x)
        utils_path = os.path.join(os.path.dirname(__file__), "utils.py")
        self.assertEqual(get_filenames(traced_code_lists), [[__file__, utils_path]])

        # === graph break occurs during inlining ===
        @torch.compile(backend=my_backend)
        def fn(x):
            z = x + 1
            y = break_it(z)
            return y * 2

        x = torch.randn(3)
        traced_code_lists = []
        fn(x)
        self.assertEqual(get_filenames(traced_code_lists), [[__file__], [utils_path]])

        # === empty graph ===
        @torch.compile(backend=my_backend)
        def fn(x):
            return x

        x = torch.randn(3)
        traced_code_lists = []
        fn(x)
        self.assertEqual(traced_code_lists, [])

    def test_add_record_function_data(self):
        with (
            mock.patch("torch.autograd.profiler._is_profiler_enabled", False),
            mock.patch("torch.autograd.profiler.record_function") as mock_rf,
        ):
            utils.CompileEventLogger.add_record_function_data(
                "test_event", key1="value1", key2="value2"
            )
            mock_rf.assert_not_called()

        with (
            mock.patch("torch.autograd.profiler._is_profiler_enabled", True),
            mock.patch("torch.autograd.profiler.record_function") as mock_rf,
        ):
            utils.CompileEventLogger.add_record_function_data("test_event")
            mock_rf.assert_not_called()

        with (
            mock.patch("torch.autograd.profiler._is_profiler_enabled", True),
            mock.patch("torch.autograd.profiler.record_function") as mock_rf,
        ):
            utils.CompileEventLogger.add_record_function_data(
                "test_event", key1="value1", key2="value2"
            )
            mock_rf.assert_called_once_with("test_event_data: key1=value1, key2=value2")

    def test_reinplace_counters_use_trigger_name_not_enum_value(self):
        """Test that ReinplaceCounters uses trigger.name in dictionary keys instead of the enum value"""
        from torch._dynamo.utils import ReinplaceCounters, ReInplaceTrigger

        # Clear any existing state
        ReinplaceCounters.clear()

        # Test with AUTO_FUNC_V1 trigger
        trigger = ReInplaceTrigger.AUTO_FUNC_V1

        # Add some values
        ReinplaceCounters.add_missed_opportunities(trigger, 2)
        ReinplaceCounters.add_missed_bytes(trigger, 512)

        # Check that the dictionary keys use the trigger name, not the enum value
        expected_tensor_key = "missed_tensors_AUTO_FUNC_V1"
        expected_bytes_key = "missed_bytes_AUTO_FUNC_V1"

        # Verify the keys exist with the correct format
        self.assertIn(
            expected_tensor_key,
            ReinplaceCounters._values,
            f"Expected key {expected_tensor_key} not found",
        )
        self.assertIn(
            expected_bytes_key,
            ReinplaceCounters._values,
            f"Expected key {expected_bytes_key} not found",
        )

        # Verify the values are correct
        self.assertEqual(ReinplaceCounters._values[expected_tensor_key], 2)
        self.assertEqual(ReinplaceCounters._values[expected_bytes_key], 512)

        # Clear for next test
        ReinplaceCounters.clear()

        # Test with a different trigger to ensure it's not hardcoded
        trigger2 = ReInplaceTrigger.TRITON_OPS
        ReinplaceCounters.add_missed_opportunities(trigger2, 3)

        expected_key2 = "missed_tensors_TRITON_OPS"
        self.assertIn(
            expected_key2,
            ReinplaceCounters._values,
            f"Expected key {expected_key2} not found",
        )
        self.assertEqual(ReinplaceCounters._values[expected_key2], 3)

        # Verify the old key doesn't exist
        self.assertNotIn("missed_tensors_AUTO_FUNC_V1", ReinplaceCounters._values)

        # Test edge case: check that we don't use the enum integer value
        # ReInplaceTrigger.AUTO_FUNC_V1 has value 1, so we verify "missed_tensors_1" doesn't exist
        self.assertNotIn(
            f"missed_tensors_{trigger.value}",
            ReinplaceCounters._values,
            "Should not use enum value (integer) in key, should use trigger.name instead",
        )


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


class TestDynamoTimed(TestCase):
    """
    Test utilities surrounding dynamo_timed.
    """

    def setUp(self):
        super().setUp()
        if hasattr(torch._dynamo, "reset_recompile_user_contexts"):
            torch._dynamo.reset_recompile_user_contexts()

    def run_forward_backward(self):
        model = torch.compile(TestModel())
        x = torch.rand([3], requires_grad=True)
        output = model(x)
        loss_fn = torch.nn.MSELoss()
        target = torch.tensor([1.0])
        loss = loss_fn(output, target)
        loss.backward()

    def warmup(self):
        # Helper to make sure any process-global lru_caches (e.g., torch_key())
        # have already executed. Just compile something.
        @torch.compile(backend="inductor")
        def add(x, y):
            return x + y

        add(torch.rand([10]), torch.rand([10]))
        utils.reset_frame_count()
        torch._logging._internal.structured_logging_overhead.clear()

    @dynamo_config.patch({"log_compilation_metrics": True})
    @inductor_config.patch({"force_disable_caches": True})
    def test_stack_trace(self):
        self.warmup()

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            self.run_forward_backward()
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        stack_trace_list = []
        for e in compilation_events:
            stack_trace_list.append(e.stack_trace)

        self.assertGreater(len(stack_trace_list), 0)
        result = "\n".join(
            item
            for sublist in stack_trace_list
            if sublist
            for item in (sublist if isinstance(sublist, list) else [sublist])
        )
        self.assertIn(
            "test_stack_trace",
            result,
            "Log file does not contain the expected string: 'test_stack_trace'",
        )

    @dynamo_config.patch({"log_compilation_metrics": True})
    @inductor_config.patch({"force_disable_caches": True})
    def test_graph_node_shapes(self):
        self.warmup()

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            self.run_forward_backward()
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]

        self.assertEqual(
            compilation_events[0].graph_node_shapes,
            "{'l_self_modules_linear_parameters_weight_': [1, 3], "
            "'l_self_modules_linear_parameters_bias_': [1], "
            "'l_x_': [3], 'linear': [1]}",
        )

    @dynamo_config.patch({"log_compilation_metrics": True})
    @inductor_config.patch({"force_disable_caches": True})
    def test_log_dynamo_start(self):
        import torch._dynamo.convert_frame as convert_frame

        self.warmup()
        self.run_forward_backward()

        # Dummy code object
        def sample_func():
            pass

        code = sample_func.__code__
        stack_strings = convert_frame.log_dynamo_start(code)
        last_entry = stack_strings[-1]
        # Check if the last entry is a valid stack trace i.e for the sample_func
        self.assertIn(
            f"Line: {code.co_firstlineno}",
            last_entry,
            "Log does not contain a Line no.",
        )
        self.assertIn(
            f"Name: {code.co_name}", last_entry, "Log does not contain a Name"
        )
        self.assertIn(
            "test_utils.py",
            last_entry,
            "Log file does not contain the expected Filename: 'test_utils.py'",
        )

        # Since the remaining logs are env specific, we just check if they are present instead of checking the exact string
        self.assertGreater(len(stack_strings), 1)

    @dynamo_config.patch({"log_compilation_metrics": True})
    @inductor_config.patch({"force_disable_caches": True})
    def test_exception_stack_trace(self):
        from torch._dynamo.exc import Unsupported

        def backward(grad_output):
            print("graph break!")  # This should trigger a Dynamo error
            return grad_output

        compiled_backward = torch.compile(backward, backend="eager", fullgraph=True)
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            with self.assertRaisesRegex(
                Unsupported,
                "Dynamo does not know how to trace builtin operator `print`",
            ):
                compiled_backward(torch.ones(3))

        compilation_events = [arg[0][0] for arg in log_event.call_args_list]

        self.assertGreater(len(compilation_events), 0)
        self.assertGreater(len(compilation_events[0].exception_stack_trace), 0)
        self.assertIn(
            "Dynamo does not know how to trace builtin operator `print`",
            compilation_events[0].exception_stack_trace[0],
            "exception_stack_trace does not contain the expected string: "
            "'Dynamo does not know how to trace builtin operator `print`'",
        )

    @dynamo_config.patch(
        {
            "log_compilation_metrics": True,
            "inline_inbuilt_nn_modules": False,
        }
    )
    @inductor_config.patch(
        {
            "bundle_triton_into_fx_graph_cache": False,
            "bundled_autotune_remote_cache": False,
            "force_disable_caches": True,
        }
    )
    # We can't easily test that timing is actually accurate. Mock time to always
    # return the same value; all durations will be zero.
    @mock.patch("time.time", return_value=0.001)
    @mock.patch("time.time_ns", return_value=100000)
    @dynamo_config.patch(specialize_float=False)
    def test_dynamo_timed(self, mock_time, mock_time_ns):
        """
        Run a compilation that includes a forward and a backward and validate
        various recorded metrics. This test could be broken into several, but the
        compilation is somewhat expensive. Instead of resetting and compiling the
        same thing multiple times, we may as well compile once and just check all
        the things that are affected by dynamo_timed.
        """
        self.warmup()

        # The logging function is different for OSS vs. internal. Let's just mock
        # and capture all the CompilationMetric objects logged.
        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            self.run_forward_backward()
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]

        def filter_expected(s: str) -> str:
            d = eval(s)
            if not dynamo_config.run_gc_after_compile:
                d.pop("gc", None)
                if "gc_time_us" in d:
                    d["gc_time_us"] = None
            return pprint.pformat(d)

        # Validate utils.compile_times(). Unfortunately, we can't test the output
        # reliably because it depends on whether 'tabulate' is installed. So we'll
        # directly inspect the dict it prints instead:
        self.assertExpectedInline(
            pprint.pformat(utils.compilation_time_metrics),
            filter_expected(
                """\
{'GraphLowering.codegen': [0.0, 0.0],
 'GraphLowering.compile_to_fn': [0.0, 0.0],
 'GraphLowering.compile_to_module': [0.0, 0.0],
 'GraphLowering.run': [0.0, 0.0],
 'OutputGraph.call_user_compiler': [0.0],
 'PyCodeCache.load_by_key_path': [0.0, 0.0],
 'PythonWrapperCodegen.generate': [0.0, 0.0],
 'Scheduler.__init__': [0.0, 0.0],
 'Scheduler.codegen': [0.0, 0.0],
 'Scheduler.fused_nodes': [0.0, 0.0],
 '_compile.compile_inner': [0.0],
 '_recursive_joint_graph_passes': [0.0],
 '_recursive_post_grad_passes': [0.0, 0.0],
 '_recursive_pre_grad_passes': [0.0],
 'additional_fake_tensor_prop': [0.0, 0.0],
 'aot_collect_metadata': [0.0],
 'aot_trace_joint_graph': [0.0],
 'async_compile.wait': [0.0, 0.0],
 'backward._backward_impl': [0.0],
 'build_guards': [0.0],
 'bytecode_tracing': [0.0],
 'compile_attempt_0': [0.0],
 'compile_file': [0.0, 0.0],
 'compile_fx.<locals>.bw_compiler': [0.0],
 'compile_fx.<locals>.fw_compiler_base': [0.0],
 'compile_fx_inner': [0.0, 0.0],
 'create_aot_dispatcher_function': [0.0],
 'fx_codegen_and_compile': [0.0, 0.0],
 'gc': [0.0],
 'insert_deferred_runtime_asserts': [0.0],
 'min_cut_rematerialization_partition': [0.0],
 'pass.joint_graph_passes.constant_fold_uniform_value': [0.0],
 'pass.joint_graph_passes.pass_pattern_0': [0.0],
 'pass.joint_graph_passes.pass_pattern_1': [0.0],
 'pass.joint_graph_passes.remove_noop_ops': [0.0],
 'pass.post_grad_passes.decompose_auto_functionalized': [0.0, 0.0],
 'pass.post_grad_passes.decompose_map_to_while_loop': [0.0, 0.0],
 'pass.post_grad_passes.decompose_scan_to_while_loop': [0.0, 0.0],
 'pass.post_grad_passes.decompose_triton_kernel_wrapper_functional': [0.0, 0.0],
 'pass.post_grad_passes.move_constructors_to_cuda': [0.0, 0.0],
 'pass.post_grad_passes.pass_pattern_0': [0.0, 0.0],
 'pass.post_grad_passes.pass_pattern_1': [0.0, 0.0],
 'pass.post_grad_passes.pass_pattern_2': [0.0, 0.0],
 'pass.post_grad_passes.post_grad_custom_pre_pass': [0.0, 0.0],
 'pass.post_grad_passes.reinplace_inplaceable_ops': [0.0, 0.0],
 'pass.post_grad_passes.remove_assert_ops': [0.0, 0.0],
 'pass.post_grad_passes.remove_noop_ops': [0.0, 0.0],
 'pass.post_grad_passes.remove_profiler_ops': [0.0, 0.0],
 'pass.post_grad_passes.stable_sort': [0.0, 0.0],
 'pass.pre_grad_passes.apply_gumbel_max_trick_pass': [0.0],
 'pass.pre_grad_passes.efficient_conv_bn_eval_pass': [0.0],
 'pass.pre_grad_passes.group_batch_fusion_passes': [0.0]}"""
                if _IS_WINDOWS
                else """\
{'GraphLowering.codegen': [0.0, 0.0],
 'GraphLowering.compile_to_fn': [0.0, 0.0],
 'GraphLowering.compile_to_module': [0.0, 0.0],
 'GraphLowering.run': [0.0, 0.0],
 'OutputGraph.call_user_compiler': [0.0],
 'PyCodeCache.load_by_key_path': [0.0, 0.0],
 'PythonWrapperCodegen.generate': [0.0, 0.0],
 'Scheduler.__init__': [0.0, 0.0],
 'Scheduler.codegen': [0.0, 0.0],
 'Scheduler.fused_nodes': [0.0, 0.0],
 '_compile.compile_inner': [0.0],
 '_recursive_joint_graph_passes': [0.0],
 '_recursive_post_grad_passes': [0.0, 0.0],
 '_recursive_pre_grad_passes': [0.0],
 'additional_fake_tensor_prop': [0.0, 0.0],
 'aot_collect_metadata': [0.0],
 'aot_trace_joint_graph': [0.0],
 'async_compile.wait': [0.0, 0.0],
 'backward._backward_impl': [0.0],
 'build_guards': [0.0],
 'bytecode_tracing': [0.0],
 'compile_attempt_0': [0.0],
 'compile_file': [0.0, 0.0],
 'compile_fx.<locals>.bw_compiler': [0.0],
 'compile_fx.<locals>.fw_compiler_base': [0.0],
 'compile_fx_inner': [0.0, 0.0],
 'create_aot_dispatcher_function': [0.0],
 'fx_codegen_and_compile': [0.0, 0.0],
 'gc': [0.0],
 'insert_deferred_runtime_asserts': [0.0],
 'min_cut_rematerialization_partition': [0.0],
 'pass.joint_graph_passes.constant_fold_uniform_value': [0.0],
 'pass.joint_graph_passes.pass_pattern_0': [0.0],
 'pass.joint_graph_passes.pass_pattern_1': [0.0],
 'pass.joint_graph_passes.remove_noop_ops': [0.0],
 'pass.post_grad_passes.decompose_auto_functionalized': [0.0, 0.0],
 'pass.post_grad_passes.decompose_map_to_while_loop': [0.0, 0.0],
 'pass.post_grad_passes.decompose_scan_to_while_loop': [0.0, 0.0],
 'pass.post_grad_passes.decompose_triton_kernel_wrapper_functional': [0.0, 0.0],
 'pass.post_grad_passes.move_constructors_to_cuda': [0.0, 0.0],
 'pass.post_grad_passes.pass_pattern_0': [0.0, 0.0],
 'pass.post_grad_passes.pass_pattern_1': [0.0, 0.0],
 'pass.post_grad_passes.pass_pattern_2': [0.0, 0.0],
 'pass.post_grad_passes.post_grad_custom_pre_pass': [0.0, 0.0],
 'pass.post_grad_passes.reinplace_inplaceable_ops': [0.0, 0.0],
 'pass.post_grad_passes.remove_assert_ops': [0.0, 0.0],
 'pass.post_grad_passes.remove_noop_ops': [0.0, 0.0],
 'pass.post_grad_passes.remove_profiler_ops': [0.0, 0.0],
 'pass.post_grad_passes.stable_sort': [0.0, 0.0],
 'pass.pre_grad_passes.apply_gumbel_max_trick_pass': [0.0],
 'pass.pre_grad_passes.efficient_conv_bn_eval_pass': [0.0],
 'pass.pre_grad_passes.group_batch_fusion_passes': [0.0]}"""
            ),  # noqa: B950
        )

        # Now validate utils.calculate_time_spent(). Formatting the return
        # value makes reading diffs much easier.
        time_spent = utils.calculate_time_spent()
        self.assertExpectedInline(
            pprint.pformat(time_spent),
            filter_expected(
                """\
{'_recursive_joint_graph_passes': 0.0,
 '_recursive_post_grad_passes': 0.0,
 '_recursive_pre_grad_passes': 0.0,
 'backend_compile': 0.0,
 'code_gen': 0.0,
 'entire_backward_compile': 0.0,
 'entire_frame_compile': 0.0,
 'gc': 0.0,
 'inductor_compile': 0.0,
 'total_wall_time': 0.0}"""
                if _IS_WINDOWS
                else """\
{'_recursive_joint_graph_passes': 0.0,
 '_recursive_post_grad_passes': 0.0,
 '_recursive_pre_grad_passes': 0.0,
 'async_compile.wait': 0.0,
 'backend_compile': 0.0,
 'code_gen': 0.0,
 'entire_backward_compile': 0.0,
 'entire_frame_compile': 0.0,
 'gc': 0.0,
 'inductor_compile': 0.0,
 'total_wall_time': 0.0}"""
            ),  # noqa: B950
        )

        # Now validate the CompilationMetrics logs. We expect a log for the
        # forward and a log for the backward.
        self.assertTrue(len(compilation_events) == 2)
        self.assertTrue(
            all(isinstance(e, utils.CompilationMetrics) for e in compilation_events)
        )

        # Remove a few fields that aren't helpful for test stability.
        for e in compilation_events:
            e.dynamo_config = None
            e.co_filename = None
            e.co_firstlineno = None
            e.inductor_config = None
            e.compiler_config = None
            e.cuda_version = None
            e.triton_version = None
            e.python_version = None
            e.pytorch_version = None
            e.stack_trace = None
            e.graph_node_shapes = None
            e.exception_stack_trace = None

        # First event is for the forward. Formatting makes reading diffs
        # much easier.
        raw = dataclasses.asdict(compilation_events[0])
        del raw["feature_usage"]
        del raw["ir_count"]
        del raw["inductor_provenance"]
        del raw["param_numel"]
        del raw["param_bytes"]
        del raw["param_count"]
        # guard_latency_us is not deterministic
        del raw["guard_latency_us"]
        self.assertExpectedInline(
            pprint.pformat(raw),
            filter_expected(
                """\
{'accumulated_cache_size': 0,
 'aot_autograd_cumulative_compile_time_us': 0,
 'aotautograd_local_cache_hit_count': 0,
 'aotautograd_local_cache_miss_count': 0,
 'aotautograd_remote_cache_hit_count': 0,
 'aotautograd_remote_cache_miss_count': 0,
 'backend_compile_time_s': 0.0,
 'backward_cumulative_compile_time_us': None,
 'cache_size': 0,
 'co_filename': None,
 'co_firstlineno': None,
 'co_name': 'forward',
 'code_gen_time_s': 0.0,
 'compile_id': '1/0',
 'compile_time_autotune_time_us': None,
 'compiler_config': None,
 'compliant_custom_ops': set(),
 'config_inline_inbuilt_nn_modules': False,
 'config_suppress_errors': False,
 'cuda_version': None,
 'cudagraph_skip_reason': None,
 'distributed_ephemeral_timeout_us': None,
 'duration_us': 0,
 'dynamo_compile_time_before_restart_us': 0,
 'dynamo_config': None,
 'dynamo_cumulative_compile_time_us': 0,
 'dynamo_time_before_restart_s': 0.0,
 'end_time_us': 100,
 'entire_frame_compile_time_s': 0.0,
 'exception_stack_trace': None,
 'fail_reason': None,
 'fail_type': None,
 'fail_user_frame_filename': None,
 'fail_user_frame_lineno': None,
 'frame_key': '1',
 'gc_time_us': 0,
 'graph_input_count': 1,
 'graph_node_count': 3,
 'graph_node_shapes': None,
 'graph_op_count': 1,
 'guard_count': 10,
 'has_guarded_code': True,
 'inductor_code_gen_cumulative_compile_time_us': 0,
 'inductor_compile_time_s': 0.0,
 'inductor_config': None,
 'inductor_cumulative_compile_time_us': 0,
 'inductor_fx_local_cache_hit_count': 0,
 'inductor_fx_local_cache_miss_count': 0,
 'inductor_fx_remote_cache_backend_type': None,
 'inductor_fx_remote_cache_hit_count': 0,
 'inductor_fx_remote_cache_hit_keys': None,
 'inductor_fx_remote_cache_miss_count': 0,
 'inductor_fx_remote_cache_miss_keys': None,
 'inline_inbuilt_nn_modules_candidate': False,
 'is_forward': True,
 'is_runtime': False,
 'joint_graph_pass_time_us': 0,
 'log_format_version': 3,
 'non_compliant_ops': set(),
 'num_graph_breaks': 0,
 'num_triton_bundles': None,
 'pgo_get_remote_code_state_time_us': None,
 'pgo_put_remote_code_state_time_us': None,
 'post_grad_pass_time_us': 0,
 'pre_grad_pass_time_us': 0,
 'python_version': None,
 'pytorch_version': None,
 'recompile_reason': None,
 'recompile_user_contexts': None,
 'remote_cache_time_saved_s': None,
 'remote_cache_version': None,
 'remote_fx_graph_cache_get_time_ms': None,
 'remote_fx_graph_cache_get_time_us': None,
 'remote_fx_graph_cache_put_time_ms': None,
 'remote_fx_graph_cache_put_time_us': None,
 'restart_reasons': set(),
 'runtime_cudagraphify_time_us': None,
 'runtime_triton_autotune_time_us': None,
 'shape_env_guard_count': 0,
 'specialize_float': False,
 'stack_trace': None,
 'start_time': 0.0001,
 'start_time_us': 100,
 'structured_logging_overhead_s': 0.0,
 'structured_logging_overhead_us': 0,
 'tensorify_float_attempt': None,
 'tensorify_float_failure': None,
 'tensorify_float_success': None,
 'triton_compile_time_us': 0,
 'triton_kernel_compile_times_us': None,
 'triton_version': None}"""
                if _IS_WINDOWS
                else """\
{'accumulated_cache_size': 0,
 'aot_autograd_cumulative_compile_time_us': 0,
 'aotautograd_local_cache_hit_count': 0,
 'aotautograd_local_cache_miss_count': 0,
 'aotautograd_remote_cache_hit_count': 0,
 'aotautograd_remote_cache_miss_count': 0,
 'backend_compile_time_s': 0.0,
 'backward_cumulative_compile_time_us': None,
 'cache_size': 0,
 'co_filename': None,
 'co_firstlineno': None,
 'co_name': 'forward',
 'code_gen_time_s': 0.0,
 'compile_id': '1/0',
 'compile_time_autotune_time_us': None,
 'compiler_config': None,
 'compliant_custom_ops': set(),
 'config_inline_inbuilt_nn_modules': False,
 'config_suppress_errors': False,
 'cuda_version': None,
 'cudagraph_skip_reason': None,
 'distributed_ephemeral_timeout_us': None,
 'duration_us': 0,
 'dynamo_compile_time_before_restart_us': 0,
 'dynamo_config': None,
 'dynamo_cumulative_compile_time_us': 0,
 'dynamo_time_before_restart_s': 0.0,
 'end_time_us': 100,
 'entire_frame_compile_time_s': 0.0,
 'exception_stack_trace': None,
 'fail_reason': None,
 'fail_type': None,
 'fail_user_frame_filename': None,
 'fail_user_frame_lineno': None,
 'frame_key': '1',
 'gc_time_us': 0,
 'graph_input_count': 1,
 'graph_node_count': 3,
 'graph_node_shapes': None,
 'graph_op_count': 1,
 'guard_count': 10,
 'has_guarded_code': True,
 'inductor_code_gen_cumulative_compile_time_us': 0,
 'inductor_compile_time_s': 0.0,
 'inductor_config': None,
 'inductor_cumulative_compile_time_us': 0,
 'inductor_fx_local_cache_hit_count': 0,
 'inductor_fx_local_cache_miss_count': 0,
 'inductor_fx_remote_cache_backend_type': None,
 'inductor_fx_remote_cache_hit_count': 0,
 'inductor_fx_remote_cache_hit_keys': None,
 'inductor_fx_remote_cache_miss_count': 0,
 'inductor_fx_remote_cache_miss_keys': None,
 'inline_inbuilt_nn_modules_candidate': False,
 'is_forward': True,
 'is_runtime': False,
 'joint_graph_pass_time_us': 0,
 'log_format_version': 3,
 'non_compliant_ops': set(),
 'num_graph_breaks': 0,
 'num_triton_bundles': None,
 'pgo_get_remote_code_state_time_us': None,
 'pgo_put_remote_code_state_time_us': None,
 'post_grad_pass_time_us': 0,
 'pre_grad_pass_time_us': 0,
 'python_version': None,
 'pytorch_version': None,
 'recompile_reason': None,
 'recompile_user_contexts': None,
 'remote_cache_time_saved_s': None,
 'remote_cache_version': None,
 'remote_fx_graph_cache_get_time_ms': None,
 'remote_fx_graph_cache_get_time_us': None,
 'remote_fx_graph_cache_put_time_ms': None,
 'remote_fx_graph_cache_put_time_us': None,
 'restart_reasons': set(),
 'runtime_cudagraphify_time_us': None,
 'runtime_triton_autotune_time_us': None,
 'shape_env_guard_count': 0,
 'specialize_float': False,
 'stack_trace': None,
 'start_time': 0.0001,
 'start_time_us': 100,
 'structured_logging_overhead_s': 0.0,
 'structured_logging_overhead_us': 0,
 'tensorify_float_attempt': None,
 'tensorify_float_failure': None,
 'tensorify_float_success': None,
 'triton_compile_time_us': 0,
 'triton_kernel_compile_times_us': None,
 'triton_version': None}"""
            ),  # noqa: B950
        )

        # Second event is for the backward
        raw = dataclasses.asdict(compilation_events[1])
        del raw["feature_usage"]
        del raw["ir_count"]
        del raw["inductor_provenance"]
        del raw["guard_latency_us"]
        del raw["param_numel"]
        del raw["param_bytes"]
        del raw["param_count"]
        self.assertExpectedInline(
            pprint.pformat(raw),
            (
                """\
{'accumulated_cache_size': None,
 'aot_autograd_cumulative_compile_time_us': None,
 'aotautograd_local_cache_hit_count': 0,
 'aotautograd_local_cache_miss_count': 0,
 'aotautograd_remote_cache_hit_count': 0,
 'aotautograd_remote_cache_miss_count': 0,
 'backend_compile_time_s': None,
 'backward_cumulative_compile_time_us': 0,
 'cache_size': None,
 'co_filename': None,
 'co_firstlineno': None,
 'co_name': None,
 'code_gen_time_s': 0.0,
 'compile_id': '1/0',
 'compile_time_autotune_time_us': None,
 'compiler_config': None,
 'compliant_custom_ops': None,
 'config_inline_inbuilt_nn_modules': False,
 'config_suppress_errors': False,
 'cuda_version': None,
 'cudagraph_skip_reason': None,
 'distributed_ephemeral_timeout_us': None,
 'duration_us': 0,
 'dynamo_compile_time_before_restart_us': None,
 'dynamo_config': None,
 'dynamo_cumulative_compile_time_us': None,
 'dynamo_time_before_restart_s': None,
 'end_time_us': 100,
 'entire_frame_compile_time_s': None,
 'exception_stack_trace': None,
 'fail_reason': None,
 'fail_type': None,
 'fail_user_frame_filename': None,
 'fail_user_frame_lineno': None,
 'frame_key': None,
 'gc_time_us': None,
 'graph_input_count': None,
 'graph_node_count': None,
 'graph_node_shapes': None,
 'graph_op_count': None,
 'guard_count': None,
 'has_guarded_code': None,
 'inductor_code_gen_cumulative_compile_time_us': 0,
 'inductor_compile_time_s': 0.0,
 'inductor_config': None,
 'inductor_cumulative_compile_time_us': 0,
 'inductor_fx_local_cache_hit_count': 0,
 'inductor_fx_local_cache_miss_count': 0,
 'inductor_fx_remote_cache_backend_type': None,
 'inductor_fx_remote_cache_hit_count': 0,
 'inductor_fx_remote_cache_hit_keys': None,
 'inductor_fx_remote_cache_miss_count': 0,
 'inductor_fx_remote_cache_miss_keys': None,
 'inline_inbuilt_nn_modules_candidate': False,
 'is_forward': False,
 'is_runtime': False,
 'joint_graph_pass_time_us': None,
 'log_format_version': 3,
 'non_compliant_ops': None,
 'num_graph_breaks': 0,
 'num_triton_bundles': None,
 'pgo_get_remote_code_state_time_us': None,
 'pgo_put_remote_code_state_time_us': None,
 'post_grad_pass_time_us': 0,
 'pre_grad_pass_time_us': None,
 'python_version': None,
 'pytorch_version': None,
 'recompile_reason': None,
 'recompile_user_contexts': None,
 'remote_cache_time_saved_s': None,
 'remote_cache_version': None,
 'remote_fx_graph_cache_get_time_ms': None,
 'remote_fx_graph_cache_get_time_us': None,
 'remote_fx_graph_cache_put_time_ms': None,
 'remote_fx_graph_cache_put_time_us': None,
 'restart_reasons': None,
 'runtime_cudagraphify_time_us': None,
 'runtime_triton_autotune_time_us': None,
 'shape_env_guard_count': None,
 'specialize_float': None,
 'stack_trace': None,
 'start_time': 0.0001,
 'start_time_us': 100,
 'structured_logging_overhead_s': 0.0,
 'structured_logging_overhead_us': 0,
 'tensorify_float_attempt': None,
 'tensorify_float_failure': None,
 'tensorify_float_success': None,
 'triton_compile_time_us': None,
 'triton_kernel_compile_times_us': None,
 'triton_version': None}"""
                if _IS_WINDOWS
                else """\
{'accumulated_cache_size': None,
 'aot_autograd_cumulative_compile_time_us': None,
 'aotautograd_local_cache_hit_count': 0,
 'aotautograd_local_cache_miss_count': 0,
 'aotautograd_remote_cache_hit_count': 0,
 'aotautograd_remote_cache_miss_count': 0,
 'backend_compile_time_s': None,
 'backward_cumulative_compile_time_us': 0,
 'cache_size': None,
 'co_filename': None,
 'co_firstlineno': None,
 'co_name': None,
 'code_gen_time_s': 0.0,
 'compile_id': '1/0',
 'compile_time_autotune_time_us': None,
 'compiler_config': None,
 'compliant_custom_ops': None,
 'config_inline_inbuilt_nn_modules': False,
 'config_suppress_errors': False,
 'cuda_version': None,
 'cudagraph_skip_reason': None,
 'distributed_ephemeral_timeout_us': None,
 'duration_us': 0,
 'dynamo_compile_time_before_restart_us': None,
 'dynamo_config': None,
 'dynamo_cumulative_compile_time_us': None,
 'dynamo_time_before_restart_s': None,
 'end_time_us': 100,
 'entire_frame_compile_time_s': None,
 'exception_stack_trace': None,
 'fail_reason': None,
 'fail_type': None,
 'fail_user_frame_filename': None,
 'fail_user_frame_lineno': None,
 'frame_key': None,
 'gc_time_us': None,
 'graph_input_count': None,
 'graph_node_count': None,
 'graph_node_shapes': None,
 'graph_op_count': None,
 'guard_count': None,
 'has_guarded_code': None,
 'inductor_code_gen_cumulative_compile_time_us': 0,
 'inductor_compile_time_s': 0.0,
 'inductor_config': None,
 'inductor_cumulative_compile_time_us': 0,
 'inductor_fx_local_cache_hit_count': 0,
 'inductor_fx_local_cache_miss_count': 0,
 'inductor_fx_remote_cache_backend_type': None,
 'inductor_fx_remote_cache_hit_count': 0,
 'inductor_fx_remote_cache_hit_keys': None,
 'inductor_fx_remote_cache_miss_count': 0,
 'inductor_fx_remote_cache_miss_keys': None,
 'inline_inbuilt_nn_modules_candidate': False,
 'is_forward': False,
 'is_runtime': False,
 'joint_graph_pass_time_us': None,
 'log_format_version': 3,
 'non_compliant_ops': None,
 'num_graph_breaks': 0,
 'num_triton_bundles': None,
 'pgo_get_remote_code_state_time_us': None,
 'pgo_put_remote_code_state_time_us': None,
 'post_grad_pass_time_us': 0,
 'pre_grad_pass_time_us': None,
 'python_version': None,
 'pytorch_version': None,
 'recompile_reason': None,
 'recompile_user_contexts': None,
 'remote_cache_time_saved_s': None,
 'remote_cache_version': None,
 'remote_fx_graph_cache_get_time_ms': None,
 'remote_fx_graph_cache_get_time_us': None,
 'remote_fx_graph_cache_put_time_ms': None,
 'remote_fx_graph_cache_put_time_us': None,
 'restart_reasons': None,
 'runtime_cudagraphify_time_us': None,
 'runtime_triton_autotune_time_us': None,
 'shape_env_guard_count': None,
 'specialize_float': None,
 'stack_trace': None,
 'start_time': 0.0001,
 'start_time_us': 100,
 'structured_logging_overhead_s': 0.0,
 'structured_logging_overhead_us': 0,
 'tensorify_float_attempt': None,
 'tensorify_float_failure': None,
 'tensorify_float_success': None,
 'triton_compile_time_us': 0,
 'triton_kernel_compile_times_us': None,
 'triton_version': None}"""
            ),  # noqa: B950
        )

    @dynamo_config.patch(
        {
            "log_compilation_metrics": True,
        }
    )
    @compiler_config.patch({"job_id": "test_job_id"})
    def test_compiler_config(self):
        def test1(x):
            return x * x

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            torch.compile(test1)(torch.randn(1))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertIn(
            '"job_id": "test_job_id"',
            compilation_events[0].compiler_config,
        )

    @dynamo_config.patch(
        {
            "log_compilation_metrics": True,
        }
    )
    def test_ir_count(self):
        # Different python versions have different potential IR counts.
        version = (sys.version_info[0], sys.version_info[1])
        self.assertIn(version, ((3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14)))
        first, second = {
            (3, 9): (10, 6),
            (3, 10): (10, 6),
            (3, 11): (11, 7),
            (3, 12): (11, 7),
            (3, 13): (11, 7),
            (3, 14): (11, 7),
        }[version]

        def test1(x):
            y = x + x
            z = y * y
            return z

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            torch.compile(test1)(torch.randn(10, 10))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(compilation_events[0].ir_count, first)

        def test2(x):
            y = x + x
            return y

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            torch.compile(test2)(torch.randn(10, 10))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(compilation_events[0].ir_count, second)

    @dynamo_config.patch(
        {
            "log_compilation_metrics": True,
        }
    )
    @inductor_config.patch(
        {"trace.enabled": True, "trace.provenance_tracking_level": 1},
    )
    def test_inductor_provenance(self):
        module = torch.nn.Linear(6, 66)
        graph_module = torch.fx.symbolic_trace(module)

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            torch.compile(graph_module)(torch.randn(6, 6))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(
            compilation_events[0].inductor_provenance,
            {'{"extern_kernels.addmm:1": []}'},
        )

    @dynamo_config.patch({"log_compilation_metrics": True})
    @inductor_config.patch({"force_disable_caches": True})
    def test_dynamic_shape_feature_use(self):
        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:

            @torch.compile()
            def f(x):
                return x * x

            f(torch.randn(4))
            f(torch.randn(3))
            compilation_events = [
                arg[0][0].feature_usage for arg in log_event.call_args_list
            ]
        self.assertIn(
            ("dynamo.automatic_dynamic_shapes", True), compilation_events[1].items()
        )

        compilation_events = []
        with (
            dynamo_config.patch({"automatic_dynamic_shapes": False}),
            mock.patch("torch._dynamo.utils.log_compilation_event") as log_event,
        ):

            @torch.compile()
            def f(x):
                return x * x

            f(torch.randn(4))
            f(torch.randn(3))
            compilation_events = [
                arg[0][0].feature_usage for arg in log_event.call_args_list
            ]
        self.assertIn(
            ("dynamo.automatic_dynamic_shapes", False), compilation_events[1].items()
        )

    @dynamo_config.patch({"log_compilation_metrics": True})
    def test_num_params(self):
        import torch.nn as nn
        import torch.nn.functional as F

        class ModelSimple(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)

            def forward(self, x):
                return F.relu(self.conv1(x))

        self.assertEqual([x.numel() for x in ModelSimple().parameters()], [500, 20])

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            m = ModelSimple()
            torch.compile(m)(torch.randn(1, 10, 10))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(compilation_events[0].param_numel, 520)
        self.assertEqual(compilation_events[0].param_bytes, 4 * 520)
        self.assertEqual(compilation_events[0].param_count, 2)

        class ModelWrapped(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m1 = ModelSimple()
                self.m2 = ModelSimple()

            def forward(self, x):
                return self.m1(x) + self.m2(x)

        compilation_events = []
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            m = ModelWrapped()
            torch.compile(m)(torch.randn(1, 10, 10))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(compilation_events[0].param_numel, 1040)
        self.assertEqual(compilation_events[0].param_bytes, 4 * 1040)
        self.assertEqual(compilation_events[0].param_count, 4)

        # Test a tied module
        l1 = nn.Linear(4, 4)
        l2 = nn.Linear(4, 4)
        m = nn.Sequential(l1, nn.Sequential(l1, l2))
        self.assertEqual([x.numel() for x in m.parameters()], [16, 4, 16, 4])
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            torch.compile(m)(torch.randn(4, 4))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(compilation_events[0].param_numel, 40)
        self.assertEqual(compilation_events[0].param_bytes, 4 * 40)
        self.assertEqual(compilation_events[0].param_count, 4)

        # Test tied weights
        l1 = nn.Linear(4, 4)
        l2 = nn.Linear(4, 4)
        l1.weight = l2.weight
        m = nn.Sequential(l1, nn.Sequential(l2))
        self.assertEqual([x.numel() for x in m.parameters()], [16, 4, 4])
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            torch.compile(m)(torch.randn(4, 4))
            compilation_events = [arg[0][0] for arg in log_event.call_args_list]
        self.assertEqual(compilation_events[0].param_numel, 24)
        self.assertEqual(compilation_events[0].param_bytes, 4 * 24)
        self.assertEqual(compilation_events[0].param_count, 3)


class TestInductorConfigParsingForLogging(TestCase):
    """
    Test for parsing inductor config for logging in CompilationMetrics.
    """

    class TestObject:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    def test_inductor_config_jsonify(self):
        """
        Sanity check if the actual inductor config is parsed correctly
        """

        inductor_config_json = utils._scrubbed_inductor_config_for_logging()
        self.assertTrue(isinstance(inductor_config_json, str))
        self.assertIn('trace"', inductor_config_json)

    @mock.patch("torch._dynamo.utils.torch._inductor.config")
    def test_inductor_config_parsing_non_conforming_items(self, mocked_inductor_config):
        """
        Test if the inductor config is parsed correctly when the config is
            - None
            - not a dict
            - not json serializable
            - complex unserializable objects
        """
        obj = TestCase
        test_mock_config = {
            "some": {"name": obj, "some": True},
            "data": {"name": obj, "some": True},
            "list": [
                {"name": obj, "some": True},
                {"name": obj, "some": True},
            ],
            "object": {
                "name": obj,
                "some": True,
                "data": {"name": obj, "some": True},
            },
        }
        expected = (
            """{"data": {"name": "Value is not JSON serializable", "some": true}, """
            """"list": [{"name": "Value is not JSON serializable", "some": true}, """
            """{"name": "Value is not JSON serializable", "some": true}], """
            """"object": {"data": {"name": "Value is not JSON serializable", "some": true}, """
            """"name": "Value is not JSON serializable", "some": true}, """
            """"some": {"name": "Value is not JSON serializable", "some": true}}"""
        )
        mocked_inductor_config.get_config_copy.return_value = test_mock_config
        inductor_config_json = utils._scrubbed_inductor_config_for_logging()
        self.assertEqual(inductor_config_json, expected)

        expected = "{}"
        mocked_inductor_config.get_config_copy.return_value = {obj: obj}
        inductor_config_json = utils._scrubbed_inductor_config_for_logging()
        self.assertEqual(inductor_config_json, expected)

        expected = "Inductor Config is not JSON serializable"
        mocked_inductor_config.get_config_copy.return_value = obj
        inductor_config_json = utils._scrubbed_inductor_config_for_logging()
        self.assertEqual(inductor_config_json, expected)

        expected = None
        mocked_inductor_config.get_config_copy.return_value = None
        inductor_config_json = utils._scrubbed_inductor_config_for_logging()
        self.assertEqual(inductor_config_json, expected)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
