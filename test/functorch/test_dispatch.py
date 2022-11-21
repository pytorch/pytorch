# Owner(s): ["module: functorch"]

from torch.testing._internal.common_utils import TestCase, run_tests

from torch._C import _dispatch_get_registrations_for_dispatch_key as get_registrations_for_dispatch_key

class TestFunctorchDispatcher(TestCase):
    def test_register_a_batching_rule_for_composite_implicit_autograd(self):
        CompositeImplicitAutograd = set(get_registrations_for_dispatch_key('CompositeImplicitAutograd'))
        FuncTorchBatched = set(get_registrations_for_dispatch_key('FuncTorchBatched'))

        overlap = CompositeImplicitAutograd & FuncTorchBatched

        self.assertEqual(len(overlap), 0)

if __name__ == '__main__':
    run_tests()
