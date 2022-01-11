import torch
from torch.testing._internal.common_utils import TestCase, run_tests
import copy

class TestPerOverloadAPI(TestCase):
    def test_basics_opoverloadpacket(self):
        # add is ony used as an example here. It is ok to update the test
        # if the semantics of add are modified in the future.
        add_packet = torch.ops.aten.add

        # class attributes
        self.assertEqual(add_packet.op_name, 'add')
        self.assertEqual(add_packet.qualified_op_name, 'aten.add')

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
        self.assertEqual(str(add_packet), "OpOverloadPacket(op='aten.add')")

        self.assertRaises(AttributeError, lambda: add_packet.foo)

    def test_basics_opoverload(self):
        add_packet = torch.ops.aten.add
        add_tensoroverload = add_packet.Tensor

        # class attributes
        self.assertEqual(add_tensoroverload.name, 'aten.add')
        self.assertEqual(add_tensoroverload.overload_name, 'Tensor')
        self.assertEqual(add_tensoroverload.overload_packet, add_packet)

        # deepcopy is a no-op
        self.assertEqual(id(add_tensoroverload), id(copy.deepcopy(add_tensoroverload)))

        # caching
        another_add_tensoroverload = torch.ops.aten.add.Tensor
        self.assertEqual(id(add_tensoroverload), id(another_add_tensoroverload))

        # pretty print
        self.assertEqual(str(add_tensoroverload), "OpOverload(op='aten.add', overload='Tensor')")

        # callable
        self.assertEqual(add_tensoroverload(torch.tensor(2), torch.tensor(3)), torch.tensor(5))

        a = torch.tensor(2)
        b = torch.tensor(0)
        torch.ops.aten.add.out(a, a, out=b)
        self.assertEqual(b, torch.tensor(4))

        self.assertRaises(RuntimeError, lambda: add_tensoroverload(a, a, out=b))

if __name__ == '__main__':
    run_tests()
