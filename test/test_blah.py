import unittest
from itertools import product
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, deviceCountAtLeast, ops)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    TestCase, run_tests, parametrize, instantiate_parametrized_tests, subtest)


class TestBlah(TestCase):
    @parametrize("x", range(5))
    def test_default_names(self, x):
        print('Passed in:', x)

    # Use default names but add an expected failure.
    @parametrize("x", [subtest(0, decorators=[unittest.expectedFailure]),
                       *range(1, 5)])
    def test_default_names_expected_failure(self, x):
        if x == 0:
            raise RuntimeError('Boom')
        print('Passed in:', x)

    @parametrize("bias", [False, True], name_fn=lambda b: 'bias' if b else 'no_bias')
    def test_custom_names(self, bias):
        print('Passed in:', bias)

    @parametrize("bias", [subtest(True, name='bias'),
                          subtest(False, name='no_bias')])
    def test_custom_names_alternate(self, bias):
        print('Passed in:', bias)

    @parametrize("x,y", [(1, 2), (1, 3), (1, 4)])
    def test_two_things_default_names(self, x, y):
        print('Passed in:', x, y)

    @parametrize("x", [1, 2, 3])
    @parametrize("y", [4, 5, 6])
    def test_two_things_composition(self, x, y):
        print('Passed in:', x, y)

    @parametrize("x", [subtest(0, decorators=[unittest.expectedFailure]),
                       *range(1, 3)])
    @parametrize("y", [4, 5, subtest(6, decorators=[unittest.expectedFailure])])
    def test_two_things_composition_expected_failure(self, x, y):
        if x == 0 or y == 6:
            raise RuntimeError('Boom')
        print('Passed in:', x, y)

    @parametrize("x", [1, 2])
    @parametrize("y", [3, 4])
    @parametrize("z", [5, 6])
    def test_three_things_composition(self, x, y, z):
        print('Passed in:', x, y, z)

    @parametrize("x", [1, 2], name_fn=str)
    @parametrize("y", [3, 4], name_fn=str)
    @parametrize("z", [5, 6], name_fn=str)
    def test_three_things_composition_custom_names(self, x, y, z):
        print('Passed in:', x, y, z)

    @parametrize("x,y", product(range(2), range(3)))
    def test_two_things_product(self, x, y):
        print('Passed in:', x, y)

    @parametrize("x,y", [subtest((1, 2), name='double'),
                         subtest((1, 3), name='triple'),
                         subtest((1, 4), name='quadruple')])
    def test_two_things_custom_names(self, x, y):
        print('Passed in:', x, y)

    @parametrize("x,y", [(1, 2), (1, 3), (1, 4)], name_fn=lambda x, y: '{}_{}'.format(x, y))
    def test_two_things_custom_names_alternate(self, x, y):
        print('Passed in:', x, y)


class TestDeviceBlah(TestCase):
    @parametrize("x", range(10))
    def test_default_names(self, device, x):
        print('Passed in:', device, x)

    @parametrize("x,y", [(1, 2), (3, 4), (5, 6)])
    def test_two_things(self, device, x, y):
        print('Passed in:', device, x, y)

    @deviceCountAtLeast(1)
    def test_multiple_devices(self, devices):
        print('Passed in:', devices)

    @ops(op_db)
    @parametrize("flag", [False, True], lambda f: 'flag_enabled' if f else 'flag_disabled')
    def test_op_parametrized(self, device, dtype, op, flag):
        print('Passed in:', device, dtype, op, flag)


instantiate_parametrized_tests(TestBlah)
instantiate_device_type_tests(TestDeviceBlah, globals())


if __name__ == '__main__':
    run_tests()
