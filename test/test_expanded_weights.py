from itertools import product

import torch
from torch.testing._internal import common_dtype
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, make_tensor, run_tests
from torch._expanded_weights import ExpandedWeight

class TestExpandedWeightFunctional(TestCase):
    def test_expanded_weight_error(self, device):
        batch_size = 3
        sample_input = make_tensor((batch_size, 4), device, torch.float32, requires_grad=True)
        sample_weight = make_tensor((4), device, torch.float32, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, r"Expanded Weights encountered but cannot handle function"):
            torch.add(sample_input, ExpandedWeight(sample_weight, batch_size))

class TestExpandedWeightAttributes(TestCase):
    def test_expanded_weight_has_attributes(self, device):
        attrs_equivalent = ['dtype', 'shape', 'requires_grad']
        attrs_special = {'grad': lambda attr, _: attr is None}
        supported_dtypes = common_dtype.floating_and_complex_types()
        for (attr, dtype) in product(attrs_equivalent, supported_dtypes):
            batch_size = 5
            orig_tensor = make_tensor((4), device, dtype)
            expanded_weight = ExpandedWeight(orig_tensor, batch_size)
            assert hasattr(expanded_weight, attr), f"Expanded Weight of type {dtype} didn't have attribute {attr}"
            actual = getattr(expanded_weight, attr)
            expected = getattr(orig_tensor, attr)
            self.assertEqual(expected, actual, f"Expected {attr} to have value {expected}, got {actual}")
        for (attr_and_func, dtype) in product(attrs_special.items(), supported_dtypes):
            attr, func = attr_and_func
            batch_size = 5
            orig_tensor = make_tensor((4), device, dtype)
            expanded_weight = ExpandedWeight(orig_tensor, batch_size)
            assert hasattr(expanded_weight, attr), f"Expanded Weight of type {dtype} didn't have attribute {attr}"
            if not func(getattr(expanded_weight, attr), orig_tensor):
                raise RuntimeError(f"{attr} got unexpected value. Was {getattr(expanded_weight, attr)}")

    def test_expanded_weight_methods(self, device):
        methods_equivalent = ['size']
        methods_special = {'__repr__': lambda attr, orig_weight: orig_weight.__repr__() in attr(),
                           '__hash__': lambda attr, orig_weight: attr() != orig_weight.__hash__()}
        supported_dtypes = common_dtype.floating_and_complex_types()
        for (method, dtype) in product(methods_equivalent, supported_dtypes):
            batch_size = 5
            orig_tensor = make_tensor((4), device, dtype)
            expanded_weight = ExpandedWeight(orig_tensor, batch_size)
            if not hasattr(expanded_weight, method):
                raise RuntimeError(f"Expanded Weight of type {dtype} didn't have method {method}")
            actual = getattr(expanded_weight, method)()
            expected = getattr(orig_tensor, method)()
            self.assertEqual(expected, actual, f"Expected {method} to have value {expected}, got {actual}")
        for (method_and_func, dtype) in product(methods_special.items(), supported_dtypes):
            method, func = method_and_func
            batch_size = 5
            orig_tensor = make_tensor((4), device, dtype)
            expanded_weight = ExpandedWeight(orig_tensor, batch_size)
            assert hasattr(expanded_weight, method), f"Expanded Weight of type {dtype} didn't have attribute {method}"
            if not func(getattr(expanded_weight, method), orig_tensor):
                raise RuntimeError(f"{method} got unexpected value. Was {getattr(expanded_weight, method)}")

instantiate_device_type_tests(TestExpandedWeightFunctional, globals())
instantiate_device_type_tests(TestExpandedWeightAttributes, globals())
if __name__ == '__main__':
    run_tests()
