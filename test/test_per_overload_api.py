# Owner(s): ["module: unknown"]
import copy
import subprocess
import sys
import textwrap

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


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

        # deepcopy is a no-op
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

        # deepcopy is a no-op
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

    def test_opoverloadpacket_init_under_sys_settrace(self):
        script = textwrap.dedent(
            """
            import sys
            import torch._ops

            saw_packet_init = False

            def trace_func(frame, event, arg):
                global saw_packet_init
                if (
                    event == "call"
                    and frame.f_code is torch._ops.OpOverloadPacket.__init__.__code__
                ):
                    saw_packet_init = True
                    packet = frame.f_locals["self"]
                    repr(packet)
                    str(packet)
                    getattr(packet, "__qualname__", None)
                    getattr(packet, "missing_overload", None)
                return trace_func

            def dummy():
                pass

            sys.settrace(trace_func)
            try:
                packet = torch._ops.OpOverloadPacket("aten::dummy", "dummy", dummy, [])
            finally:
                sys.settrace(None)

            assert saw_packet_init
            assert str(packet) == "aten.dummy"
            """
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            self.fail("OpOverloadPacket initialization hung under sys.settrace")
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_decompose(self):
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        self.assertEqual(
            torch.ops.aten.linear.default.decompose(x, y),
            torch.ops.aten.linear.default(x, y),
        )


if __name__ == "__main__":
    run_tests()
