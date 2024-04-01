import copy

import torch


def get_wrapped_fn(fn):
    if hasattr(fn, "__wrapped__"):
        wrapped = fn.__wrapped__
        return get_wrapped_fn(wrapped)
    else:
        return fn


class XPUPatch:
    def __enter__(self):
        self.onlyCUDA_fn = torch.testing._internal.common_device_type.onlyCUDA
        torch.testing._internal.common_device_type.onlyCUDA = (
            torch.testing._internal.common_device_type.onlyXPU
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.testing._internal.common_device_type.onlyCUDA = self.onlyCUDA_fn


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
    # Device test classes need to be wrapped as nested classes. Otherwise, they are not
    # visible to other test files because instantiate_device_type_tests
    # will change its name and its test member functions.
    # So the generic_base_class here should be a Nested and Wrapped class.
    #  ex:
    #    class TestCommon(TestCase):
    #      ...
    #
    #    class Namespace:
    #      class TestCommonWrapper(TestCommon):
    #        ...
    # We need to wrap TestCommon as TestCommonWrapper to avoid instantiate_device_type_tests
    # changing its meta information and moving it under Namespace class to prevent test runners
    # from picking it up and running it.
    base_class, *_ = generic_base_class.__bases__
    generic_base_class_members = set(base_class.__dict__.keys()) - set(
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
            test = getattr(base_class, name)
            setattr(generic_test_class, name, copy.deepcopy(test))
        else:  # Ports non-test member
            nontest = getattr(base_class, name)
            setattr(generic_test_class, name, nontest)
