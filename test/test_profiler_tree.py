# Owner(s): ["oncall: profiler"]

import functools
import os
import re
import textwrap
import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase, run_tests, IS_WINDOWS, TEST_WITH_CROSSREF)


class ProfilerTree:

    @staticmethod
    def test(f):
        """Mark unit test that will be using ProfilerTree to test traces.

        This decorator serves two purposes. First, it provides a method name
        that `format` can use to tell where the test runner (which is
        environment specific) ends and the unit test begins. Second, it runs
        the test with replicates and allows `assertTreesMatch` to adjust
        based on which replicate is running.
        """

        @functools.wraps(f)
        def begin_unit_test_marker(self, replicates=5):
            try:
                for i in range(replicates):
                    self.tree_replicate = i
                    return f(self)
            finally:
                delattr(self, "tree_replicate")
        return begin_unit_test_marker

    @classmethod
    def format(cls, profiler, indent: int = 0):

        def flatten(nodes, depth=0, out=None):
            if out is None:
                out = []

            for node in nodes:
                out.append((depth, cls.fmt_name(node.name())))
                flatten(node.children, depth + 1, out)

            return out

        flat_nodes = flatten(profiler.kineto_results.experimental_event_tree())
        min_depth = min([d + 1 for d, name in flat_nodes if "begin_unit_test_marker" in name] or [0])
        return textwrap.indent(
            "\n".join([f"{'  ' * (d - min_depth)}{name.rstrip()}" for d, name in flat_nodes if d >= min_depth]),
            " " * indent)

    @staticmethod
    def fmt_name(name: str) -> str:
        # torch::autograd::Node relies on c10::demangle to generate names, and
        # Windows demangles to include `struct` in the name.
        if IS_WINDOWS:
            name = name.replace('struct torch::autograd::AccumulateGrad', 'torch::autograd::AccumulateGrad')

        match = re.match(r"(.*)\.py\(([0-9]+)\): (.*)$", name)
        if match:
            filename, lineno, fn = match.groups()

            # This test can appear as `test/test_profiler_tree.py` depending on
            # where it is run from.
            if filename.endswith(os.path.splitext(__file__)[0]):
                filename = os.path.split(os.path.splitext(__file__)[0])[1]

            # We test against a string literal, so all paths have to look like POSIX paths.
            filename = filename.replace(os.sep, "/")

            # We don't want to have to update this test every time PyTorch changes.
            lineno = lineno if os.path.split(filename.strip())[1] == "test_profiler_tree" else "..."
            return f"{filename}.py({lineno}): {fn}"

        return re.sub(
            "object at 0x[0-9a-fA-F]+>",
            "object at 0xXXXXXXXXXXXX>",
            name)

class TestProfilerTree(TestCase):
    def assertTreesMatch(self, actual: str, expected: str):
        # Warning: Here be dragons
        #   Different platforms will have subtly different behavior for Python
        #   tracing. Observed differences include:
        #     1) Windows symbolicates names differently from posix
        #     2) The profile callback for c_call does not fire for Tensor.__pow__
        #        on certain platforms. This is not caused by the function tracer,
        #        but by cPython itself.
        #
        # The purpose of these unit tests is to ensure that the profiler is
        # doing reasonable things. When these platform dependent variations occur
        # simply coerce them into a platform independent form. If you made a
        # change in the codebase which changes the trace produced, simply use
        # EXPECTTEST_ACCEPT=1 to update the tests to reflect the new structure.

        replicate = getattr(self, "tree_replicate", None)
        self.assertIsNotNone(replicate, "Please annotate test with `@ProfilerTree.test`")

        # The profiler should produce deterministic results and should return
        # to a clean state after each run. As a result, only the first
        # replicate is allowed to update `expected`. If subsequent runs do not
        # match it is a bug in the profiler.
        if replicate:
            self.assertEqual(actual, expected)
        else:
            self.assertExpectedInline(actual, expected, skip=1)

    @ProfilerTree.test
    def test_profiler_experimental_tree(self):
        t1, t2 = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        with torch.profiler.profile() as p:
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = (y - z) ** 2
            loss.backward()

        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            aten::add
            aten::ones
              aten::empty
              aten::fill_
            aten::sub
            aten::pow
              aten::result_type
              aten::to
            aten::ones_like
              aten::empty_like
                aten::empty_strided
              aten::fill_
            autograd::engine::evaluate_function: PowBackward0
              PowBackward0
                aten::pow
                  aten::result_type
                  aten::to
                  aten::copy_
                aten::mul
                  aten::mul
                    aten::to
                      aten::_to_copy
                        aten::empty_strided
                        aten::copy_
                aten::mul
            autograd::engine::evaluate_function: SubBackward0
              SubBackward0
                aten::neg
            autograd::engine::evaluate_function: AddBackward0
              AddBackward0
            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad
              torch::autograd::AccumulateGrad
                aten::new_empty_strided
                  aten::empty_strided
                aten::copy_
            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad
              torch::autograd::AccumulateGrad
                aten::detach
                  detach"""
        )

    @ProfilerTree.test
    def test_profiler_experimental_tree_with_record_function(self):
        with torch.profiler.profile() as p:
            with torch.autograd.profiler.record_function("Top level Annotation"):
                with torch.autograd.profiler.record_function("First Annotation"):
                    x = torch.ones((1,), requires_grad=True)

                # Check that we correctly handle the case when a user
                # annotation does not call `__exit__`.
                _ = torch.autograd.profiler.record_function("Second Annotation").__enter__()

                y = x + 1
                with torch.autograd.profiler.record_function("Third Annotation"):
                    y.backward()

        # NB: The `aten::zeros` before the record function annotations are due to
        # `at::cpp_custom_type_hack`. When we switch to `torch::CustomClassHolder`
        # they will disappear.
        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            aten::zeros
              aten::zeros
                aten::empty
                aten::zero_
            Top level Annotation
              aten::empty
              aten::zeros
                aten::zeros
                  aten::empty
                  aten::zero_
              First Annotation
                aten::empty
                aten::ones
                  aten::empty
                  aten::fill_
              aten::zeros
                aten::zeros
                  aten::empty
                  aten::zero_
              Second Annotation
                aten::empty
                aten::add
                  aten::to
                    aten::_to_copy
                      aten::empty_strided
                      aten::copy_
                aten::zeros
                  aten::zeros
                    aten::empty
                    aten::zero_
                Third Annotation
                  aten::empty
                  aten::ones_like
                    aten::empty_like
                      aten::empty_strided
                    aten::fill_
                  autograd::engine::evaluate_function: AddBackward0
                    AddBackward0
                  autograd::engine::evaluate_function: torch::autograd::AccumulateGrad
                    torch::autograd::AccumulateGrad
                      aten::new_empty_strided
                        aten::empty_strided
                      aten::copy_"""
        )

    @ProfilerTree.test
    def test_profiler_experimental_tree_with_memory(self):
        t1, t2 = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        with torch.profiler.profile(profile_memory=True) as p:
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = (y - z) ** 2
            loss.backward()

        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            aten::add
              [memory]
            aten::ones
              aten::empty
                [memory]
              aten::fill_
            aten::sub
              [memory]
            aten::pow
              aten::result_type
              aten::to
              [memory]
            aten::ones_like
              aten::empty_like
                aten::empty_strided
                  [memory]
              aten::fill_
            autograd::engine::evaluate_function: PowBackward0
              PowBackward0
                aten::pow
                  aten::result_type
                  aten::to
                  [memory]
                  aten::copy_
                aten::mul
                  [memory]
                  aten::mul
                    aten::to
                      aten::_to_copy
                        aten::empty_strided
                          [memory]
                        aten::copy_
                    [memory]
                    [memory]
                  [memory]
                aten::mul
                  [memory]
                [memory]
                [memory]
              [memory]
            autograd::engine::evaluate_function: SubBackward0
              SubBackward0
                aten::neg
                  [memory]
              [memory]
            autograd::engine::evaluate_function: AddBackward0
              AddBackward0
            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad
              torch::autograd::AccumulateGrad
                aten::new_empty_strided
                  aten::empty_strided
                    [memory]
                aten::copy_
            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad
              torch::autograd::AccumulateGrad
                aten::detach
                  detach
            [memory]"""
        )

    @unittest.skipIf(TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite.")
    @unittest.skipIf(torch.has_cuda, "CUDA invokes extra Python functions.")
    @ProfilerTree.test
    def test_profiler_experimental_tree_with_memory_and_stack(self):
        t1, t2 = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        with torch.profiler.profile(with_stack=True, profile_memory=True) as p:
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = torch.pow(y - z, 2)
            loss.backward()

        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            test_profiler_tree.py(304): test_profiler_experimental_tree_with_memory_and_stack
              torch/profiler/profiler.py(...): __enter__
                torch/profiler/profiler.py(...): start
                  torch/profiler/profiler.py(...): _transit_action
                    torch/profiler/profiler.py(...): start_trace
                      torch/autograd/profiler.py(...): _start_trace
                      <built-in method kineto_available of PyCapsule object at 0xXXXXXXXXXXXX>
                      torch/profiler/profiler.py(...): _get_distributed_info
                        torch/distributed/__init__.py(...): is_available
                          <built-in function hasattr>
                        torch/distributed/distributed_c10d.py(...): is_initialized
              <built-in method add of type object at 0xXXXXXXXXXXXX>
                aten::add
                  [memory]
              <built-in method ones of type object at 0xXXXXXXXXXXXX>
                aten::ones
                  aten::empty
                    [memory]
                  aten::fill_
              aten::sub
                [memory]
              <built-in method pow of type object at 0xXXXXXXXXXXXX>
                aten::pow
                  aten::result_type
                  aten::to
                  [memory]
              torch/_tensor.py(...): backward
                <built-in function _has_torch_function_unary>
                torch/autograd/__init__.py(...): backward
                  <built-in function isinstance>
                  <built-in function isinstance>
                  <built-in function len>
                  torch/autograd/__init__.py(...): _tensor_or_tensors_to_tuple
                  torch/autograd/__init__.py(...): _make_grads
                    <built-in function isinstance>
                    <built-in method numel of Tensor object at 0xXXXXXXXXXXXX>
                    <built-in method ones_like of type object at 0xXXXXXXXXXXXX>
                      aten::ones_like
                        aten::empty_like
                          aten::empty_strided
                            [memory]
                        aten::fill_
                    <built-in method numel of Tensor object at 0xXXXXXXXXXXXX>
                  <built-in method numel of Tensor object at 0xXXXXXXXXXXXX>
                    autograd::engine::evaluate_function: PowBackward0
                      PowBackward0
                        aten::pow
                          aten::result_type
                          aten::to
                          [memory]
                          aten::copy_
                        aten::mul
                          [memory]
                          aten::mul
                            aten::to
                              aten::_to_copy
                                aten::empty_strided
                                  [memory]
                                aten::copy_
                            [memory]
                            [memory]
                          [memory]
                        aten::mul
                          [memory]
                        [memory]
                        [memory]
                      [memory]
                    autograd::engine::evaluate_function: SubBackward0
                      SubBackward0
                        aten::neg
                          [memory]
                      [memory]
                    autograd::engine::evaluate_function: AddBackward0
                      AddBackward0
                    autograd::engine::evaluate_function: torch::autograd::AccumulateGrad
                      torch::autograd::AccumulateGrad
                        aten::new_empty_strided
                          aten::empty_strided
                            [memory]
                        aten::copy_
                    autograd::engine::evaluate_function: torch::autograd::AccumulateGrad
                      torch::autograd::AccumulateGrad
                        aten::detach
                          detach
                [memory]
              torch/profiler/profiler.py(...): __exit__
                torch/profiler/profiler.py(...): stop
                  torch/profiler/profiler.py(...): _transit_action
                    <built-in method numel of Tensor object at 0xXXXXXXXXXXXX>
                      enum.py(...): __hash__
                        <built-in function hash>
                    torch/profiler/profiler.py(...): stop_trace
                      torch/autograd/profiler.py(...): __exit__
                        <built-in method _disable_profiler of PyCapsule object at 0xXXXXXXXXXXXX>"""
        )

    @unittest.skipIf(TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite.")
    @unittest.skipIf(torch.has_cuda, "CUDA invokes extra Python functions.")
    @ProfilerTree.test
    def test_profiler_experimental_tree_with_stack_and_modules(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [
                    torch.nn.ReLU(),
                    torch.nn.Linear(1, 1),
                    torch.nn.ReLU(),
                ]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for l in self.layers:
                    x = l(x)
                return x

        model = MyModule()
        with torch.profiler.profile(with_stack=True) as p:
            for _ in range(2):
                model(torch.ones((1,)))
        self.maxDiff = None
        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            test_profiler_tree.py(428): test_profiler_experimental_tree_with_stack_and_modules
              torch/profiler/profiler.py(...): __enter__
                torch/profiler/profiler.py(...): start
                  torch/profiler/profiler.py(...): _transit_action
                    torch/profiler/profiler.py(...): start_trace
                      torch/autograd/profiler.py(...): _start_trace
                      <built-in method kineto_available of PyCapsule object at 0xXXXXXXXXXXXX>
                      torch/profiler/profiler.py(...): _get_distributed_info
                        torch/distributed/__init__.py(...): is_available
                          <built-in function hasattr>
                        torch/distributed/distributed_c10d.py(...): is_initialized
              <built-in method ones of type object at 0xXXXXXXXXXXXX>
                aten::ones
                  aten::empty
                  aten::fill_
              nn.Module: MyModule_0
                <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                test_profiler_tree.py(422): forward
                  nn.Module: ReLU_0
                    <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                    torch/nn/modules/activation.py(...): forward
                      torch/nn/functional.py(...): relu
                        <built-in function _has_torch_function_unary>
                        <built-in method relu of type object at 0xXXXXXXXXXXXX>
                          aten::relu
                            aten::clamp_min
                  nn.Module: Linear_0
                    <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                    torch/nn/modules/linear.py(...): forward
                      torch/nn/modules/module.py(...): __getattr__
                      torch/nn/modules/module.py(...): __getattr__
                      <built-in function linear>
                        aten::linear
                          aten::t
                            aten::transpose
                              aten::as_strided
                          aten::matmul
                            aten::t
                              aten::transpose
                                aten::as_strided
                            aten::mv
                              aten::empty
                              aten::addmv_
                          aten::add_
                  nn.Module: ReLU_1
                    <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                    torch/nn/modules/activation.py(...): forward
                      torch/nn/functional.py(...): relu
                        <built-in function _has_torch_function_unary>
                        <built-in method relu of type object at 0xXXXXXXXXXXXX>
                          aten::relu
                            aten::clamp_min
              <built-in method ones of type object at 0xXXXXXXXXXXXX>
                aten::ones
                  aten::empty
                  aten::fill_
              nn.Module: MyModule_0
                <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                test_profiler_tree.py(422): forward
                  nn.Module: ReLU_0
                    <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                    torch/nn/modules/activation.py(...): forward
                      torch/nn/functional.py(...): relu
                        <built-in function _has_torch_function_unary>
                        <built-in method relu of type object at 0xXXXXXXXXXXXX>
                          aten::relu
                            aten::clamp_min
                  nn.Module: Linear_0
                    <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                    torch/nn/modules/linear.py(...): forward
                      torch/nn/modules/module.py(...): __getattr__
                      torch/nn/modules/module.py(...): __getattr__
                      <built-in function linear>
                        aten::linear
                          aten::t
                            aten::transpose
                              aten::as_strided
                          aten::matmul
                            aten::t
                              aten::transpose
                                aten::as_strided
                            aten::mv
                              aten::empty
                              aten::addmv_
                          aten::add_
                  nn.Module: ReLU_1
                    <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                    torch/nn/modules/activation.py(...): forward
                      torch/nn/functional.py(...): relu
                        <built-in function _has_torch_function_unary>
                        <built-in method relu of type object at 0xXXXXXXXXXXXX>
                          aten::relu
                            aten::clamp_min
              torch/profiler/profiler.py(...): __exit__
                torch/profiler/profiler.py(...): stop
                  torch/profiler/profiler.py(...): _transit_action
                    <built-in method get of dict object at 0xXXXXXXXXXXXX>
                      enum.py(...): __hash__
                        <built-in function hash>
                    torch/profiler/profiler.py(...): stop_trace
                      torch/autograd/profiler.py(...): __exit__
                        <built-in method _disable_profiler of PyCapsule object at 0xXXXXXXXXXXXX>"""
        )

if __name__ == '__main__':
    run_tests()
