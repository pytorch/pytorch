# Owner(s): ["module: dynamo"]
import dataclasses
import pprint
from unittest import mock

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
from torch._dynamo import utils
from torch._inductor.test_case import TestCase


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
                a * 6,
                fp64_ref=fp64_ref,
                use_larger_multiplier_for_smaller_tensor=True,
                tol=tol,
            )
        )

    def test_larger_multiplier_for_even_smaller_tensor(self):
        """
        Tesnor numel <=10
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
        @torch.compile
        def add(x, y):
            return x + y

        add(torch.rand([10]), torch.rand([10]))
        utils.reset_frame_count()

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
        }
    )
    # We can't easily test that timing is actually accurate. Mock time to always
    # return the same value; all durations will be zero.
    @mock.patch("time.time", return_value=0.001)
    @mock.patch("time.time_ns", return_value=100000)
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

        # Validate utils.compile_times(). Unfortunately, we can't test the output
        # reliably because it depends on whether 'tabulate' is installed. So we'll
        # directly inspect the dict it prints instead:
        self.assertExpectedInline(
            pprint.pformat(utils.compilation_time_metrics),
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
 'async_compile.wait': [0.0, 0.0],
 'backward._backward_impl': [0.0],
 'compile_file': [0.0, 0.0],
 'compile_fx.<locals>.bw_compiler': [0.0],
 'compile_fx.<locals>.fw_compiler_base': [0.0],
 'compile_fx_inner': [0.0, 0.0],
 'create_aot_dispatcher_function': [0.0]}""",  # noqa: B950
        )

        # Now validate utils.calculate_time_spent(). Formatting the return
        # value makes reading diffs much easier.
        time_spent = utils.calculate_time_spent()
        self.assertExpectedInline(
            pprint.pformat(time_spent),
            """\
{'_recursive_joint_graph_passes': 0.0,
 '_recursive_post_grad_passes': 0.0,
 '_recursive_pre_grad_passes': 0.0,
 'backend_compile': 0.0,
 'code_gen': 0.0,
 'entire_backward_compile': 0.0,
 'entire_frame_compile': 0.0,
 'inductor_compile': 0.0,
 'total_wall_time': 0.0}""",  # noqa: B950
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

        # First event is for the forward. Formatting makes reading diffs
        # much easier.
        self.assertExpectedInline(
            pprint.pformat(dataclasses.asdict(compilation_events[0])),
            """\
{'accumulated_cache_size': 0,
 'aot_autograd_cumulative_compile_time_us': 0,
 'backend_compile_time_s': 0.0,
 'backward_cumulative_compile_time_us': None,
 'cache_size': 0,
 'co_filename': None,
 'co_firstlineno': None,
 'co_name': 'forward',
 'code_gen_time_s': 0.0,
 'compile_id': '1/0',
 'compliant_custom_ops': [],
 'config_inline_inbuilt_nn_modules': False,
 'config_suppress_errors': False,
 'cuda_synchronize_time_us': None,
 'distributed_ephemeral_timeout_us': None,
 'duration_us': 0,
 'dynamo_compile_time_before_restart_us': 0,
 'dynamo_config': None,
 'dynamo_cumulative_compile_time_us': 0,
 'dynamo_time_before_restart_s': 0.0,
 'end_time_us': 100,
 'entire_frame_compile_time_s': 0.0,
 'fail_reason': None,
 'fail_type': None,
 'fail_user_frame_filename': None,
 'fail_user_frame_lineno': None,
 'frame_key': '1',
 'graph_input_count': 1,
 'graph_node_count': 3,
 'graph_op_count': 1,
 'guard_count': 8,
 'has_guarded_code': True,
 'inductor_code_gen_cumulative_compile_time_us': 0,
 'inductor_compile_time_s': 0.0,
 'inductor_config': None,
 'inductor_cumulative_compile_time_us': 0,
 'inductor_fx_remote_cache_backend_type': None,
 'inductor_fx_remote_cache_hit_count': None,
 'inductor_fx_remote_cache_hit_keys': None,
 'inductor_fx_remote_cache_miss_count': None,
 'inductor_fx_remote_cache_miss_keys': None,
 'is_forward': True,
 'joint_graph_pass_time_us': 0,
 'log_format_version': 3,
 'non_compliant_ops': [],
 'num_triton_bundles': None,
 'post_grad_pass_time_us': 0,
 'pre_grad_pass_time_us': 0,
 'remote_cache_time_saved_s': None,
 'remote_cache_version': None,
 'remote_fx_graph_cache_get_time_ms': None,
 'remote_fx_graph_cache_get_time_us': None,
 'remote_fx_graph_cache_put_time_ms': None,
 'remote_fx_graph_cache_put_time_us': None,
 'restart_reasons': [],
 'runtime_cudagraphify_time_us': None,
 'runtime_triton_autotune_time_us': None,
 'shape_env_guard_count': 0,
 'specialize_float': True,
 'start_time': 0.0001,
 'start_time_us': 100,
 'structured_logging_overhead_s': 0.0,
 'structured_logging_overhead_us': 0,
 'triton_compile_time_us': None}""",  # noqa: B950
        )

        # Second event is for the backward
        self.assertExpectedInline(
            pprint.pformat(dataclasses.asdict(compilation_events[1])),
            """\
{'accumulated_cache_size': None,
 'aot_autograd_cumulative_compile_time_us': None,
 'backend_compile_time_s': None,
 'backward_cumulative_compile_time_us': 0,
 'cache_size': None,
 'co_filename': None,
 'co_firstlineno': None,
 'co_name': None,
 'code_gen_time_s': 0.0,
 'compile_id': '1/0',
 'compliant_custom_ops': None,
 'config_inline_inbuilt_nn_modules': None,
 'config_suppress_errors': None,
 'cuda_synchronize_time_us': None,
 'distributed_ephemeral_timeout_us': None,
 'duration_us': 0,
 'dynamo_compile_time_before_restart_us': None,
 'dynamo_config': None,
 'dynamo_cumulative_compile_time_us': None,
 'dynamo_time_before_restart_s': None,
 'end_time_us': 100,
 'entire_frame_compile_time_s': None,
 'fail_reason': None,
 'fail_type': None,
 'fail_user_frame_filename': None,
 'fail_user_frame_lineno': None,
 'frame_key': None,
 'graph_input_count': None,
 'graph_node_count': None,
 'graph_op_count': None,
 'guard_count': None,
 'has_guarded_code': None,
 'inductor_code_gen_cumulative_compile_time_us': 0,
 'inductor_compile_time_s': 0.0,
 'inductor_config': None,
 'inductor_cumulative_compile_time_us': 0,
 'inductor_fx_remote_cache_backend_type': None,
 'inductor_fx_remote_cache_hit_count': None,
 'inductor_fx_remote_cache_hit_keys': None,
 'inductor_fx_remote_cache_miss_count': None,
 'inductor_fx_remote_cache_miss_keys': None,
 'is_forward': False,
 'joint_graph_pass_time_us': None,
 'log_format_version': 3,
 'non_compliant_ops': None,
 'num_triton_bundles': None,
 'post_grad_pass_time_us': 0,
 'pre_grad_pass_time_us': None,
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
 'start_time': 0.0001,
 'start_time_us': 100,
 'structured_logging_overhead_s': 0.0,
 'structured_logging_overhead_us': 0,
 'triton_compile_time_us': None}""",  # noqa: B950
        )


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
            "some": {1: "0", obj: "this", "name": obj, "some": True},
            "data": {1: "0", obj: "this", "name": obj, "some": True},
            "list": [
                {1: "0", obj: "this", "name": obj, "some": True},
                {1: "0", obj: "this", "name": obj, "some": True},
            ],
            "object": {
                1: "0",
                obj: "this",
                "name": obj,
                "some": True,
                "data": {1: "0", obj: "this", "name": obj, "some": True},
            },
        }
        expected = (
            """{"some": {"1": "0", "name": "Value is not JSON serializable", "some": true},"""
            """ "data": {"1": "0", "name": "Value is not JSON serializable", "some": true}, "list": """
            """[{"1": "0", "name": "Value is not JSON serializable", "some": true}, """
            """{"1": "0", "name": "Value is not JSON serializable", "some": true}], "object": """
            """{"1": "0", "name": "Value is not JSON serializable", "some": true, "data": """
            """{"1": "0", "name": "Value is not JSON serializable", "some": true}}}"""
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
