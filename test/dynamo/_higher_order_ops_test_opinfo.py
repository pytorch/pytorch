# Owner(s): ["module: dynamo"]
# flake8: noqa: F403,F405

try:
    from ._higher_order_ops_test_utils import *
except ImportError:
    from _higher_order_ops_test_utils import *


xfail_hops_compile = {
    # aot_eager
    "map",  # assert type(args[1].realize()) is TensorVariable
    "scan",  # scan is not an OpOverload
    "local_map_hop",  # can't retrace
    # inductor
    "while_loop",  # LoweringException: AssertionError
    "flex_attention",  # LoweringException: AssertionError
}


class TestHigherOrderOpsOpInfo(torch._dynamo.test_case.TestCaseWithNestedGraphBreaks):
    @requires_cuda_and_triton
    @parametrize("backend", ("aot_eager", "inductor"))
    @ops(
        list(filter(lambda op: op.name not in xfail_hops_compile, hop_db)),
        allowed_dtypes=(torch.float,),
    )
    def test_hops_compile(self, device, dtype, op, backend):
        # Ensure HOPs can be compiled

        if backend == "aot_eager" and op.name == "invoke_quant":
            raise unittest.SkipTest(
                "TODO: partitioner fails. migrate canonicalization to aot eager backend"
            )

        sample_inputs_itr = op.sample_inputs(
            device, dtype, requires_grad=op.supports_autograd
        )
        for inp in sample_inputs_itr:
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            eager_args = (*input, *inp.args)
            eager_kwargs = inp.kwargs
            compiled_args = deepcopy(eager_args)
            compiled_kwargs = deepcopy(eager_kwargs)

            def fn(args, kwargs):
                return op.op(*args, **(kwargs))

            compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)

            eager_out = fn(eager_args, eager_kwargs)
            compiled_out = compiled_fn(compiled_args, compiled_kwargs)
            self.assertEqual(eager_out, compiled_out)
