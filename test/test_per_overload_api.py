# Owner(s): ["module: unknown"]
import copy
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.library import _scoped_library
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode


def _run_concurrently(fn, num_threads=8):
    """Runs fn in num_threads threads at once and returns their results.

    The threads start together and switch as often as possible, which makes
    them likely to interleave inside lazily initialized code.
    """
    barrier = threading.Barrier(num_threads)

    def run():
        barrier.wait()
        return fn()

    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(num_threads) as executor:
            futures = [executor.submit(run) for _ in range(num_threads)]
            return [future.result() for future in futures]
    finally:
        sys.setswitchinterval(switch_interval)


class _RecordOps(TorchDispatchMode):
    """Records the ops that it intercepts and returns their first argument."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func)
        return args[0]


class TestPerOverloadAPI(TestCase):
    def test_basics_opoverloadpacket(self):
        # add is only used as an example here. It is ok to update the test
        # if the semantics of add are modified in the future.
        add_packet = torch.ops.aten.add

        # class attributes
        self.assertEqual(add_packet.__name__, "add")
        self.assertEqual(str(add_packet), "aten.add")

        # callable
        self.assertEqual(add_packet(torch.tensor(2), torch.tensor(3)), torch.tensor(5))

        # correct module
        self.assertEqual(add_packet.__module__, add_packet.op.__module__)

        # caching
        another_add_packet = torch.ops.aten.add
        self.assertEqual(id(add_packet), id(another_add_packet))

        # copy and deepcopy are no-ops
        self.assertEqual(id(add_packet), id(copy.copy(add_packet)))
        self.assertEqual(id(add_packet), id(copy.deepcopy(add_packet)))

        # pretty print
        self.assertEqual(repr(add_packet), "<OpOverloadPacket(op='aten.add')>")

        self.assertRaises(AttributeError, lambda: add_packet.foo)

    def test_basics_opoverload(self):
        add_packet = torch.ops.aten.add
        add_tensoroverload = add_packet.Tensor

        # class attributes
        self.assertEqual(str(add_tensoroverload), "aten.add.Tensor")
        self.assertEqual(add_tensoroverload.__name__, "add.Tensor")
        self.assertEqual(add_tensoroverload.overloadpacket, add_packet)

        # copy and deepcopy are no-ops
        self.assertEqual(id(add_tensoroverload), id(copy.copy(add_tensoroverload)))
        self.assertEqual(id(add_tensoroverload), id(copy.deepcopy(add_tensoroverload)))

        # caching
        another_add_tensoroverload = torch.ops.aten.add.Tensor
        self.assertEqual(id(add_tensoroverload), id(another_add_tensoroverload))

        # pretty print
        self.assertEqual(
            repr(add_tensoroverload), "<OpOverload(op='aten.add', overload='Tensor')>"
        )

        # callable
        self.assertEqual(
            add_tensoroverload(torch.tensor(2), torch.tensor(3)), torch.tensor(5)
        )

        a = torch.tensor(2)
        b = torch.tensor(0)
        torch.ops.aten.add.out(a, a, out=b)
        self.assertEqual(b, torch.tensor(4))

        self.assertRaises(RuntimeError, lambda: add_tensoroverload(a, a, out=b))

    def test_copy_higher_order_operator(self):
        cond = torch.ops.higher_order.cond
        self.assertIs(copy.copy(cond), cond)
        self.assertIs(copy.deepcopy(cond), cond)

    def test_traced_original_aten_is_the_op(self):
        # FX copies node metadata such as "original_aten" with copy.copy; the
        # op recorded on a traced node must still be the singleton op.
        from functorch.compile import aot_function, nop

        original_atens = []

        def fw_compiler(gm, example_inputs):
            original_atens.extend(
                node.meta.get("original_aten")
                for node in gm.graph.nodes
                if node.op == "call_function"
            )
            return gm

        fn = aot_function(torch.sin, fw_compiler=fw_compiler, bw_compiler=nop)
        fn(torch.randn(3, requires_grad=True))
        self.assertIn(torch.ops.aten.sin.default, original_atens)

    def test_decompose(self):
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        self.assertEqual(
            torch.ops.aten.linear.default.decompose(x, y),
            torch.ops.aten.linear.default(x, y),
        )

    @parametrize("target", ["namespace", "packet", "overload", "dispatch"])
    def test_concurrent_first_access(self, target):
        # Threads racing on the first access to objects in torch.ops must all
        # get the objects that torch.ops ends up with. For "dispatch", these are
        # the OpOverloads passed to __torch_dispatch__ the first time an op is
        # dispatched to Python, which C++ caches without holding a reference to
        # them; a duplicate would be freed while C++ still points to it.
        prefix = f"_test_concurrent_first_access_{target}"
        names = [f"{prefix}{i}" for i in range(200)]
        with _scoped_library(prefix, "DEF") as lib:
            for name in names:
                lib.define(f"{name}(Tensor x) -> Tensor")
            ns = getattr(torch.ops, prefix)

            def get_objects():
                if target == "namespace":
                    return [getattr(torch.ops, name) for name in names]
                if target == "packet":
                    return [getattr(ns, name) for name in names]
                return [getattr(ns, name).default for name in names]

            def access_objects():
                if target != "dispatch":
                    return get_objects()
                x = torch.ones(1)
                with _RecordOps() as mode:
                    for name in names:
                        getattr(ns, name)(x)
                return mode.ops

            try:
                for objects in _run_concurrently(access_objects):
                    for obj, expected in zip(objects, get_objects(), strict=True):
                        self.assertIs(obj, expected)
            finally:
                # Remove the namespaces that "namespace" creates.
                for name in names:
                    if vars(torch.ops).pop(name, None) is not None:
                        torch.ops._dir.remove(name)


instantiate_parametrized_tests(TestPerOverloadAPI)


if __name__ == "__main__":
    run_tests()
