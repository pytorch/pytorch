# Owner(s): ["oncall: fx"]

import unittest

import torch

from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.constructor_mover_pass import (
    ConstructorMoverPass,
    ZeroOrMultipleDevicesError,
)
from torch.fx.passes.infra.pass_base import PassResult
from torch.testing._internal.common_utils import run_tests, TestCase

DEVICE_KEY = "device"
CUDA = "cuda"
CPU = "cpu"


@unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
class TestConstructorMoverPass(TestCase):
    # Check whether the type of the device keyword-argument of the
    # module nodes whose targets are listed in moved_targets is
    # equal to the device (argument).
    def _check_device_for_targets(self, gm, device, moved_targets):
        for n in gm.graph.nodes:
            if n.target in moved_targets:
                self.assertEqual(device, n.kwargs[DEVICE_KEY].type)

    # Actually runs the pass and checks whether we returned the
    # expected value based on inplace argument.
    def _run_pass(self, gm, inplace=False, allow_outputs=False) -> PassResult:
        r_ = ConstructorMoverPass(CUDA, inplace=inplace, allow_outputs=allow_outputs)(
            gm
        )
        self.assertIsNotNone(r_)

        r: PassResult = r_  # type: ignore[arg-type]
        if inplace:
            self.assertEqual(id(gm), id(r.graph_module))

        return r

    # Move the GraphModule constructors to CUDA, and check if the
    # moved_targets actually were moved.
    def _move_and_check(self, gm, moved_targets, inplace=False, modified=True):
        r = self._run_pass(gm, inplace)
        r_gm, r_modified = r
        self.assertEqual(modified, r_modified)
        self._check_device_for_targets(r_gm, CUDA, moved_targets)

    # Trace fn and return a corresponding GraphModule.
    def _get_graph_module(self, fn, args):
        gm = make_fx(fn, tracing_mode="fake")(*args)
        return gm

    # Transform fn into a GraphModule and run the checks out/in-place.
    def _check(self, fn, args, moved_targets, modified=True):
        gm = self._get_graph_module(fn, args)
        self._check_device_for_targets(gm, CPU, moved_targets)
        self._move_and_check(gm, moved_targets, inplace=False, modified=modified)
        self._move_and_check(gm, moved_targets, inplace=True, modified=modified)

    def test_no_movable_constructors(self):
        # Constructor is not movable, because it's an output.
        def foo():
            return torch.arange(5)

        self._check(foo, tuple(), {}, modified=False)

    def test_no_target_devices(self):
        # Since the pass is executed with allow_outputs=True, we consider the
        # only constructor in the function as movable.
        #
        # ZeroOrMultipleDevicesError is raised because there's no CUDA device
        # being used inside the function.
        def foo():
            return torch.arange(5)

        gm = self._get_graph_module(foo, tuple())

        with self.assertRaises(ZeroOrMultipleDevicesError):
            self._run_pass(gm, allow_outputs=True)

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires 2 cuda devices")
    def test_multiple_target_devices(self):
        # ZeroOrMultipleDevicesError is raised because there are 2 CUDA tensors
        # each with a different CUDA device in the same program.
        # TODO: can be supported if the CPU constructor is not used together
        # with each of them.
        def foo(x, y):
            return x[torch.arange(5)], y

        x0 = torch.rand(5, device="cuda:0")
        x1 = torch.rand(5, device="cuda:1")
        gm = self._get_graph_module(foo, (x0, x1))

        with self.assertRaises(ZeroOrMultipleDevicesError):
            self._run_pass(gm)

    def test_index_single_cpu_constructor(self):
        # The indice constructor is moved, since:
        #   - it's not the function's output
        #   - index.Tensor is allowed to be mixed
        def foo(x):
            return x[torch.arange(5)]

        x = torch.ones(5, device=CUDA)
        self._check(foo, (x,), {torch.ops.aten.arange.default})

    def test_index_multiple_cpu_constructors(self):
        # Same as test_index_single_cpu_constructor, but with multiple
        # cpu tensors as index.
        def foo(x):
            i0 = torch.arange(2)
            i1 = torch.arange(2)
            i2 = torch.arange(2)
            return x[i0, i1, i2]

        x = torch.ones((4, 4, 4), device=CUDA)
        self._check(foo, (x,), {torch.ops.aten.arange.default})

    def test_index_multiple_cpu_constructors_with_raw_output(self):
        # Same as test_index_multiple_cpu_constructors, but we return
        # one of the indices. Since we don't execute the pass with
        # allowed_outputs=True, i0 can't be moved. Making all other
        # arguments of index.Tensor not movable.
        # TODO: i1 and i2 can be moved even if i0 can't.
        def foo(x):
            i0 = torch.arange(2)
            i1 = torch.arange(2)
            i2 = torch.arange(2)
            return x[i0, i1, i2], i0

        x = torch.ones((4, 4, 4), device=CUDA)
        self._check(foo, (x,), {}, modified=False)


if __name__ == "__main__":
    run_tests()
