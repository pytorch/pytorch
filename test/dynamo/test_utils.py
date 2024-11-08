# Owner(s): ["module: dynamo"]
import pprint
from unittest import mock

import torch
from torch._dynamo import config, utils
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
    Test the utilities surrounding dynamo_timed.
    """

    def setUp(self):
        super().setUp()
        utils.reset_frame_count()
        torch._dynamo.reset()

        # We can't easily test that timing is actually correct. Mock time to always
        # return the same value and all runtimes will be zero.
        time_patch = mock.patch("time.time", return_value=0.001)
        time_patch.start()
        self.addCleanup(time_patch.stop)

        time_ns_patch = mock.patch("time.time_ns", return_value=100000)
        time_ns_patch.start()
        self.addCleanup(time_ns_patch.stop)

    def run_forward_backward(self):
        model = torch.compile(TestModel())
        x = torch.rand([3], requires_grad=True)
        output = model(x)
        loss_fn = torch.nn.MSELoss()
        target = torch.tensor([1.0])
        loss = loss_fn(output, target)
        loss.backward()

    def test_compile_times(self):
        self.run_forward_backward()
        res = utils.compile_times()
        self.assertExpectedInline(
            res,
            """\
TorchDynamo compilation metrics:
Function                              Runtimes (s)
------------------------------------  --------------
_compile.compile_inner                0.0000
OutputGraph.call_user_compiler        0.0000
_recursive_pre_grad_passes            0.0000
create_aot_dispatcher_function        0.0000
compile_fx.<locals>.fw_compiler_base  0.0000
compile_fx_inner                      0.0000, 0.0000
_recursive_post_grad_passes           0.0000, 0.0000
GraphLowering.run                     0.0000, 0.0000
GraphLowering.compile_to_module       0.0000, 0.0000
Scheduler.__init__                    0.0000, 0.0000
Scheduler.codegen                     0.0000, 0.0000
PythonWrapperCodegen.generate         0.0000, 0.0000
PyCodeCache.load_by_key_path          0.0000, 0.0000
async_compile.wait                    0.0000, 0.0000
compile_file                          0.0000, 0.0000
compile_fx.<locals>.bw_compiler       0.0000""",  # noqa: B950
        )

    def test_calculate_time_spent(self):
        self.run_forward_backward()
        res = utils.calculate_time_spent()
        # Formatting makes reading diffs much easier.
        self.assertExpectedInline(
            pprint.pformat(res),
            """\
{'backend_compile': 0.0,
 'code_gen': 0.0,
 'entire_frame_compile': 0.0,
 'inductor_compile': 0.0,
 'total_wall_time': 0.0}""",  # noqa: B950
        )

    @config.patch("log_compilation_metrics", True)
    def test_log_compilation_event(self):
        # Mock the internal logging function and just capture all the
        # CompilationMetric objects it sees.
        with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
            self.run_forward_backward()
            events = [arg[0][0] for arg in log_event.call_args_list]

        # We expect log for the forward and a log for the backward.
        self.assertTrue(len(events) == 2)
        self.assertTrue(all(isinstance(e, utils.CompilationMetrics) for e in events))

        # Remove some fields that aren't helpful for test stability.
        for e in events:
            e.dynamo_config = None
            e.co_filename = None
            e.co_firstlineno = None

        # First event is for the forward. Formatting makes reading diffs
        # much easier.
        self.assertExpectedInline(
            pprint.pformat(events[0]),
            """\
CompilationMetrics(compile_id='0/0',
                   frame_key='1',
                   co_name='forward',
                   co_filename=None,
                   co_firstlineno=None,
                   cache_size=0,
                   accumulated_cache_size=0,
                   guard_count=33,
                   shape_env_guard_count=0,
                   graph_op_count=1,
                   graph_node_count=5,
                   graph_input_count=3,
                   start_time=0.0001,
                   entire_frame_compile_time_s=0.0,
                   backend_compile_time_s=0.0,
                   inductor_compile_time_s=0.0,
                   code_gen_time_s=0.0,
                   fail_type=None,
                   fail_reason=None,
                   fail_user_frame_filename=None,
                   fail_user_frame_lineno=None,
                   non_compliant_ops=set(),
                   compliant_custom_ops=set(),
                   restart_reasons=set(),
                   dynamo_time_before_restart_s=0.0,
                   has_guarded_code=True,
                   remote_cache_time_saved_s=0,
                   structured_logging_overhead_s=0.0,
                   config_suppress_errors=False,
                   config_inline_inbuilt_nn_modules=True,
                   specialize_float=True,
                   dynamo_config=None,
                   is_forward=True,
                   num_triton_bundles=None,
                   remote_fx_graph_cache_get_time_ms=None,
                   remote_fx_graph_cache_put_time_ms=None,
                   start_time_us=100,
                   duration_us=0,
                   dynamo_cumulative_compile_time_us=0,
                   aot_autograd_cumulative_compile_time_us=0,
                   inductor_cumulative_compile_time_us=0,
                   inductor_code_gen_cumulative_compile_time_us=0,
                   triton_compile_time_us=None,
                   runtime_cudagraphify_time_us=None,
                   runtime_triton_autotune_time_us=None,
                   dynamo_compile_time_before_restart_us=0,
                   cuda_synchronize_time_us=None,
                   distributed_ephemeral_timeout_us=0,
                   structured_logging_overhead_us=0,
                   remote_fx_graph_cache_get_time_us=None,
                   remote_fx_graph_cache_put_time_us=None)""",  # noqa: B950
        )

        # Second event is for the backward
        self.assertExpectedInline(
            pprint.pformat(events[1]),
            """\
CompilationMetrics(compile_id='0/0',
                   frame_key=None,
                   co_name=None,
                   co_filename=None,
                   co_firstlineno=None,
                   cache_size=None,
                   accumulated_cache_size=None,
                   guard_count=None,
                   shape_env_guard_count=None,
                   graph_op_count=None,
                   graph_node_count=None,
                   graph_input_count=None,
                   start_time=None,
                   entire_frame_compile_time_s=None,
                   backend_compile_time_s=None,
                   inductor_compile_time_s=0.0,
                   code_gen_time_s=0.0,
                   fail_type=None,
                   fail_reason=None,
                   fail_user_frame_filename=None,
                   fail_user_frame_lineno=None,
                   non_compliant_ops=None,
                   compliant_custom_ops=None,
                   restart_reasons=None,
                   dynamo_time_before_restart_s=None,
                   has_guarded_code=None,
                   remote_cache_time_saved_s=None,
                   structured_logging_overhead_s=0.0,
                   config_suppress_errors=None,
                   config_inline_inbuilt_nn_modules=None,
                   specialize_float=None,
                   dynamo_config=None,
                   is_forward=False,
                   num_triton_bundles=None,
                   remote_fx_graph_cache_get_time_ms=None,
                   remote_fx_graph_cache_put_time_ms=None,
                   start_time_us=100,
                   duration_us=0,
                   dynamo_cumulative_compile_time_us=None,
                   aot_autograd_cumulative_compile_time_us=None,
                   inductor_cumulative_compile_time_us=0,
                   inductor_code_gen_cumulative_compile_time_us=0,
                   triton_compile_time_us=None,
                   runtime_cudagraphify_time_us=None,
                   runtime_triton_autotune_time_us=None,
                   dynamo_compile_time_before_restart_us=None,
                   cuda_synchronize_time_us=None,
                   distributed_ephemeral_timeout_us=None,
                   structured_logging_overhead_us=0,
                   remote_fx_graph_cache_get_time_us=None,
                   remote_fx_graph_cache_put_time_us=None)""",  # noqa: B950
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
