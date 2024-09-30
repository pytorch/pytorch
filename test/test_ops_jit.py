# Owner(s): ["module: unknown"]

from functools import partial
from textwrap import dedent

import torch
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_jit import (
    check_against_reference,
    JitCommonTestCase,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    clone_input_helper,
    first_sample,
    IS_SANDCASTLE,
    run_tests,
    TestCase,
    unMarkDynamoStrictTest,
)
from torch.testing._internal.jit_metaprogramming_utils import (
    check_alias_annotation,
    create_script_fn,
    create_traced_fn,
)
from torch.testing._internal.jit_utils import (
    disable_autodiff_subgraph_inlining,
    is_lambda,
)


# variant testing is only done with torch.float and torch.cfloat to avoid
#   excessive test times and maximize signal to noise ratio
_variant_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float, torch.cfloat)
)


# Tests operators for consistency between JIT and eager, also checks
#   correctness of JIT specific alias schemas and intended
#   autodifferentiation behavior.
# Inherits from JitCommonTestCase instead of TestCase directly to share
#   functionality with original test_jit.py method operator tests
@unMarkDynamoStrictTest
class TestJit(JitCommonTestCase):
    exact_dtype = True

    # Tests that the forward and backward passes of operations produce the
    #   same values for the cross-product of op variants (function, method, inplace)
    #   and runtimes (eager, traced, scripted).
    # TODO WARNING: inplace x {traced, scripted} not currently tested
    @_variant_ops(op_db)
    def test_variant_consistency_jit(self, device, dtype, op):
        _requires_grad = dtype in op.supported_backward_dtypes(
            torch.device(device).type
        )

        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex
        samples = op.sample_inputs(
            device,
            dtype,
            requires_grad=_requires_grad,
            include_conjugated_inputs=include_conjugated_inputs,
        )

        # Acquires variants to test
        func = op.get_op()
        method = op.get_method()
        variants = {
            # TODO: inplace tests currently fail, fix and add inplace variant
            "function": func,
            "method": method,
        }

        # scripting strips the torch.ops prefix from these operators
        # incorrectly; don't bother testing this case.  Count this
        # as "testing"
        if isinstance(func, torch._ops.OpOverload):
            self.skipTest("variant consistency doesn't work on torch.ops")

        # TODO: find better way to standardize on op registration itself..
        has_fake_function = op.name in ["resize_", "resize_as_"]

        if has_fake_function:
            variants = {"method": getattr(torch.Tensor, op.name)}
            samples = op.sample_inputs(device, dtype, requires_grad=False)

        tested = False
        for sample in samples:
            # Test traced and scripted consistency
            for func_type, variant in variants.items():
                if variant is None:
                    continue

                # scripting and check_alias_analysis do not work with lambdas
                # lambdas are typically used as a way to simulate methods without
                # functional variants, so rely on the other variant for testing
                # for now
                if is_lambda(variant):
                    continue

                tested = True
                try:
                    self.indiv_variant_test_jit(
                        device, dtype, op, sample, func_type, variant, has_fake_function
                    )
                except Exception as e:
                    variant_error_info = dedent(
                        f"""
                        Error testing {op.name} {func_type} variant
                        with dtype: {dtype}
                        with inputs {sample}:
                    """
                    )
                    raise Exception(variant_error_info) from e  # noqa: TRY002

        assert tested, "JIT Test does not execute any logic"

    def indiv_variant_test_jit(
        self, device, dtype, op, sample, func_type, variant, has_fake_function
    ):
        _requires_grad = dtype in op.supported_backward_dtypes(
            torch.device(device).type
        )
        support_script = op.supports_scripting
        # Create accessor for script function variant
        name = op.name + "_" if func_type == "inplace" else op.name

        # run with disable_autodiff_subgraph_inlining(True) to test
        #   autodiff support. Context manager forces the graph to contain
        #   DifferentiableGraph nodes if they are present
        with disable_autodiff_subgraph_inlining():
            # Check scripted forward, grad, and grad grad
            if support_script:
                script_fn = create_script_fn(self, name, func_type)

            def out_fn(output):
                # Processes the output for autograd
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output

            def get_sample():
                return (
                    clone_input_helper(sample.input)
                    if op.name[-1] == "_"
                    else sample.input
                )

            if support_script:
                check_against_reference(
                    self,
                    script_fn,
                    op.get_op(),
                    out_fn,
                    (get_sample(),) + sample.args,
                    sample.kwargs,
                    no_grad=not _requires_grad,
                    no_gradgrad=not op.supports_gradgrad,
                )

            # Check traced forward, grad, and grad grad
            # TODO: fix tracing here
            supports_tracing = op.supports_tracing and not has_fake_function
            if op.assert_jit_shape_analysis:
                self.assertTrue(supports_tracing)

            if supports_tracing:
                traced_fn = create_traced_fn(self, variant)
                check_against_reference(
                    self,
                    traced_fn,
                    op.get_op(),
                    out_fn,
                    (get_sample(),) + sample.args,
                    sample.kwargs,
                    no_grad=not _requires_grad,
                    no_gradgrad=not op.supports_gradgrad,
                )

            # Check alias annotation schema for correctness (make
            #   sure inputs that aren't supposed to be modified aren't)
            # Note: only runs in float32 because schema isn't affected by dtype,
            #   so running it on all dtypes is would be excessive
            if dtype == torch.float32:
                # TODO: no reason why we cant run this with tracing graph
                if support_script and op.name != "rsub":
                    check_alias_annotation(
                        name,
                        (get_sample(),) + sample.args,
                        sample.kwargs,
                        func_type=func_type,
                        aten_name=op.aten_name,
                    )

                # TODO: use script graph as well
                checked_shape_analysis = False
                if supports_tracing:
                    out = variant(get_sample(), *sample.args, **sample.kwargs)

                    # right now, tuple of outputs and tensor output supported
                    # TODO: list of tensor outputs
                    tuple_of_tensors = isinstance(out, tuple) and all(
                        isinstance(elem, torch.Tensor) for elem in out
                    )

                    if isinstance(out, torch.Tensor) or tuple_of_tensors:
                        if tuple_of_tensors:
                            sizes = [elem.size() for elem in out]
                        else:
                            sizes = out.size()
                        self.checkShapeAnalysis(
                            sizes, traced_fn.graph, op.assert_jit_shape_analysis
                        )
                        checked_shape_analysis = True
                if op.assert_jit_shape_analysis:
                    self.assertTrue(checked_shape_analysis)

            # Check autodifferentiation of nodes for traced and scripted graphs, only need to check once per sample
            if dtype is torch.float32:
                # Sandcastle doesn't fuse nodes
                if IS_SANDCASTLE:
                    # fusible nodes are expected to be found in FusionGroups in the DifferentiableGraphs
                    nonfusible_nodes = (
                        op.autodiff_nonfusible_nodes + op.autodiff_fusible_nodes
                    )
                    fusible_nodes = []
                else:
                    nonfusible_nodes = op.autodiff_nonfusible_nodes
                    fusible_nodes = op.autodiff_fusible_nodes

                if supports_tracing:
                    self.assertAutodiffNode(
                        traced_fn.last_graph,
                        op.assert_autodiffed,
                        nonfusible_nodes,
                        fusible_nodes,
                    )
                if support_script:
                    self.assertAutodiffNode(
                        script_fn.last_graph,
                        op.assert_autodiffed,
                        nonfusible_nodes,
                        fusible_nodes,
                    )

    # alias testing is only done with torch.float for the same reason
    _alias_ops = partial(ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float,))

    @_alias_ops(op for op in op_db if op.aliases)
    def test_jit_alias_remapping(self, device, dtype, op):
        # NOTE: only tests on first sample
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        sample = first_sample(self, samples)

        # [Scripting Data Preparation]
        # Prepare data for test scripting
        # Below we prepare strings of args/kwargs with and without type annotations.
        # These strings are inserted into function template strings which is then torch scripted.
        # - args string is ["t0"] corresponding to the "input" tensor required by the op
        # - args_kw is the value of args and strings of kwargs used to call the op (without type annotations), for example,
        # ["to", "1.0", "(1,)", "True", "tensor(1.0)"] -> def fn(t0): return variant(t0, 1.0, (1,), True, tensor(1.0))
        args = ["t0"]

        def quote_strs(v):
            if isinstance(v, str):
                return f"'{v}'"

            return str(v)

        args_kw = (
            args
            + [f"{v}" for v in sample.args]
            + [f"{k}={quote_strs(v)}" for k, v in sample.kwargs.items()]
        )

        # Prepare data for test tracing
        sample_args_kwargs = ()
        if len(sample.args) > 0:
            sample_args_kwargs += (sample.args,)
        if len(sample.kwargs) > 0:
            sample_args_kwargs += (sample.kwargs,)

        original_name = op.aten_name
        original_name_inplace = original_name + "_"
        expected_dtype = op(sample.input, *sample.args, **sample.kwargs).dtype

        for a_op in op.aliases:
            inplace = a_op.inplace_variant
            method_or_inplace = [a_op.inplace_variant, a_op.method_variant]
            variants = (
                v
                for v in (a_op.op, a_op.method_variant, a_op.inplace_variant)
                if v is not None
            )

            # Test scripting:
            for variant in variants:
                variant_name = variant.__name__
                op_name = original_name_inplace if variant is inplace else original_name

                if variant in method_or_inplace:
                    fn_template = """
                        def _fn(t0{c}):
                            return t0.{alias_name}({args_kw})
                    """
                    # remove the first input tensor
                    script = fn_template.format(
                        c=", " if len(args_kw[1:]) > 1 else "",
                        args_kw=", ".join(args_kw[1:]),
                        alias_name=variant_name,
                    )
                else:
                    fn_template = """
                        def _fn({args}):
                            return variant({args_kw})
                    """
                    script = fn_template.format(
                        args=", ".join(args),
                        args_kw=", ".join(args_kw),
                    )

                # Required to avoid undefined value: tensor error in JIT
                # compilation of the function template
                script = script.replace("tensor(", "torch.tensor(")

                scripted = torch.jit.CompilationUnit(script)._fn

                if variant is inplace and not torch.can_cast(expected_dtype, dtype):
                    try:
                        inp = clone_input_helper(sample.input)
                        scripted(inp)
                    except Exception as e:
                        continue
                    self.fail(
                        "Inplace operation on integer tensor that should be promoted to float didn't fail!"
                    )

                inp = clone_input_helper(sample.input)
                scripted(inp)
                inp = clone_input_helper(sample.input)
                graph = scripted.graph_for(inp)
                FileCheck().check(op.aten_name).check_not(variant_name).run(graph)

            # Test tracing:
            for variant in variants:
                variant_name = variant.__name__
                op_name = original_name_inplace if variant is inplace else original_name

                def _fn(*sample_args, **sample_kwargs):
                    return variant(*sample_args, **sample_kwargs)

                inp = (clone_input_helper(sample.input),) + sample_args_kwargs
                traced = torch.jit.trace(_fn, *inp)
                inp = (clone_input_helper(sample.input),) + sample_args_kwargs
                traced(*inp)
                inp = (clone_input_helper(sample.input),) + sample_args_kwargs
                graph = traced.graph_for(*inp)
                FileCheck().check(op_name).check_not(variant_name).run(graph)


instantiate_device_type_tests(TestJit, globals())

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
