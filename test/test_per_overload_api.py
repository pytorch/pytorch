# Owner(s): ["module: unknown"]
import copy

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


if __name__ == "__main__":
    run_tests()
