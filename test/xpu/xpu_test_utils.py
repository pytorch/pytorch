import copy
import os
import sys

import torch


def get_wrapped_fn(fn):
    if hasattr(fn, "__wrapped__"):
        wrapped = fn.__wrapped__
        return get_wrapped_fn(wrapped)
    else:
        return fn


def DO_NOTHING(*args, **kwargs):
    # Do nothing
    pass


class XPUPatchForImport:
    def __init__(self) -> None:
        current_file_path = os.path.realpath(__file__)
        self.test_package = os.path.dirname(os.path.dirname(current_file_path))
        self.original_path = sys.path.copy()
        self.test_case_cls = torch.testing._internal.common_utils.TestCase
        self.only_cuda_fn = torch.testing._internal.common_device_type.onlyCUDA
        self.instantiate_fn = (
            torch.testing._internal.common_device_type.instantiate_device_type_tests
        )

    def __enter__(self):
        torch.testing._internal.common_device_type.instantiate_device_type_tests = (
            DO_NOTHING
        )
        torch.testing._internal.common_utils.TestCase = (
            torch.testing._internal.common_utils.NoTest
        )
        torch.testing._internal.common_device_type.onlyCUDA = (
            torch.testing._internal.common_device_type.onlyXPU
        )
        sys.path.append(self.test_package)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path = self.original_path
        torch.testing._internal.common_device_type.instantiate_device_type_tests = (
            self.instantiate_fn
        )
        torch.testing._internal.common_device_type.onlyCUDA = self.only_cuda_fn
        torch.testing._internal.common_utils.TestCase = self.test_case_cls


# Copy the test cases from generic_base_class to generic_test_class.
# It serves to reuse test cases. Regarding some newly added hardware,
# they have to copy and paste code manually from some test files to reuse the test
# cases. The maintenance effort is non-negligible as the test file changing is
# always on its way.
# This function provides an auto mechanism by replacing manual copy-paste w/
# automatically copying the test member functions from the base class to the dest test
# class.
def copy_tests(
    generic_test_class, generic_base_class, applicable_list=None, bypass_list=None
):
    assert len(generic_base_class.__bases__) > 0
    generic_base_class_members = set(generic_base_class.__dict__.keys()) - set(
        generic_test_class.__dict__.keys()
    )
    assert not (
        applicable_list and bypass_list
    ), "Does not support setting both applicable list and bypass list."
    if applicable_list:
        generic_base_class_members = [
            x for x in generic_base_class_members if x in applicable_list
        ]
    if bypass_list:
        generic_base_class_members = [
            x for x in generic_base_class_members if x not in bypass_list
        ]

    generic_base_tests = [x for x in generic_base_class_members if x.startswith("test")]

    for name in generic_base_class_members:
        if name in generic_base_tests:  # Instantiates test member
            test = getattr(generic_base_class, name)
            setattr(generic_test_class, name, copy.deepcopy(test))
        else:  # Ports non-test member
            nontest = getattr(generic_base_class, name)
            setattr(generic_test_class, name, nontest)
