import functools
import unittest

import torch

import torch.utils._pytree as pytree

from torch._dynamo.utils import clone_input
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch.overrides import TorchFunctionMode
from torch.testing._internal.optests import (
    aot_autograd_check,
    autograd_registration_check,
    fake_check,
)


def safe_schema_check(op, args, kwargs):
    args, kwargs = pytree.tree_map(clone_input, (args, kwargs))
    with SchemaCheckMode():
        result = op(*args, **kwargs)
        return result


def safe_autograd_registration_check(op, args, kwargs):
    # Don't perform autograd_registration_check if none of the inputs require grad.
    if not pytree.tree_any_only(
        torch.Tensor, lambda x: x.requires_grad, (args, kwargs)
    ):
        return
    args, kwargs = pytree.tree_map(clone_input, (args, kwargs))
    return autograd_registration_check(op, args, kwargs)


def safe_fake_check(op, args, kwargs):
    args, kwargs = pytree.tree_map(clone_input, (args, kwargs))
    return fake_check(op, args, kwargs, dynamic_only=False)


def safe_aot_autograd_check(op, args, kwargs, dynamic):
    def func(*args, **kwargs):
        args, kwargs = pytree.tree_map_only(torch.Tensor, torch.clone, (args, kwargs))
        return op(*args, **kwargs)

    # aot_autograd_check runs func(*args, **kwargs) multiple times
    # and assumes `func` does not modify its inputs.
    return aot_autograd_check(func, args, kwargs, dynamic, check_gradients="auto")


# Test util requirements
# - The test util must have signature (op: OpOverload, args, kwargs)
# - The test util must NOT mutate args, kwargs.
# - The test utils in this list must not be prefixes of each other. For example,
#   having both "test_schema" and "test_schema_is_functional" is NOT OK.
ALL_TEST_UTILS = {
    "test_schema": safe_schema_check,
    "test_autograd_registration": safe_autograd_registration_check,
    "test_faketensor": safe_fake_check,
    "test_aot_dispatch_static": functools.partial(
        safe_aot_autograd_check,
        dynamic=False,
    ),
    "test_aot_dispatch_dynamic": functools.partial(
        safe_aot_autograd_check,
        dynamic=True,
    ),
}


def generate_opcheck_tests(
    testcase, namespaces, failures_dict, additional_decorators, test_utils
):
    """Given an existing TestCase, use the existing tests to generate
    additional validation tests for custom operators.

    For {all existing tests in the TestCase} x {all test utils},
    we will generate one new test. The new test runs a TorchFunctionMode
    that intercepts ``op(*args, **kwargs)`` calls and invokes
    ``test_util(op, *args, **kwargs)``, where ``op`` is an operator.

    The test_util that we support are in ALL_TEST_UTILS. They are:
    - test_schema: This runs SchemaCheckMode.
    - test_autograd_registration: This runs autograd_registration_check.
    - test_faketensor: This runs CrossRefFakeMode.
    - test_aot_dispatch_static: This runs aot_autograd_check, which:
        checks that the outputs (and gradients, if they are computable)
        are the same under eager-mode PyTorch and using AOTAutograd.
    - test_aot_dispatch_dynamic: Same as aot_dispatch_static, but
        runs AOTAutograd using dynamic shapes instead of static shapes.

    The generated test will have name ``{test_util}__{original_name}``.
    For example, if there is a method named ``test_cumsum``, then
    we will generate a ``test_schema__test_cumsum``,
    ``test_faketensor__test_cumsum``, etc.

    Args:
        testcase: The testcase we will modify and generate additional tests for.
        namespaces: We will only intercept calls to custom operators with these
                    namespaces.
        failures_dict: See ``validate_failures_dict`` for more details
        additional_decorators: Pass us some decorators
        test_utils: a list of test_utils to generate. Example: ["test_schema", "test_faketensor"]
    """
    if not issubclass(testcase, unittest.TestCase):
        raise ValueError(
            f"Expected testcase to be subclass of unittest.TestCase, got {type(testcase)}"
        )
    test_methods = [
        m
        for m in dir(testcase)
        if m.startswith("test_") and callable(getattr(testcase, m))
    ]
    validate_failures_dict(failures_dict, test_utils, testcase)

    def construct_method(attr, prefix, tester):
        method = getattr(testcase, attr)
        new_method_name = prefix + "__" + attr

        def new_method(*args, **kwargs):
            with OpCheckMode(namespaces, tester, failures_dict, new_method_name):
                result = method(*args, **kwargs)
            return result

        if new_method_name in additional_decorators:
            for dec in additional_decorators[new_method_name]:
                new_method = dec(new_method)

        if hasattr(testcase, new_method_name):
            raise RuntimeError(
                f"Tried to autogenerate {new_method_name} but {testcase} already "
                f"has method named {new_method_name}. Please rename the original "
                f"method on the TestCase."
            )
        setattr(testcase, new_method_name, new_method)

    test_utils = {name: ALL_TEST_UTILS[name] for name in test_utils}
    for attr in test_methods:
        for prefix, tester in test_utils.items():
            construct_method(attr, prefix, tester)


TEST_OPTIONS = ("xfail", "skip", "success")


def validate_failures_dict(failure_dict, test_utils, testcase):
    """Validates the failures dict.

    The failure dict looks something like the following.
    It maps operator name (qualname) to a list of autogenerated tests.
    Each autogenerated test may have a check for the operator (if the operator is
    called by the test); the dictionary specifies if we should skip the check,
    or if we expect some check to fail.

    {
        "fbgemm::split_lengths": {
            "test_schema__test_split_lengths": "xfail",
            "test_schema__test_split_lengths_empty": "skip",
        }
        "fbgemm::gather_lengths": {
            "test_schema__test_gather_lengths": "xfail",
        }
    }

    We require that all keys are sorted in alphabetical order. This makes
    it easier for us to codemod the failures_dict.
    """
    qualnames = list(failure_dict.keys())
    if qualnames != sorted(qualnames):
        raise RuntimeError("The failures dict must be sorted in alphabetical order")
    for qualname, test_to_option in failure_dict.items():
        test_names = list(test_to_option.keys())
        if test_names != sorted(test_names):
            raise RuntimeError(
                f"failures_dict['{qualname}']'s keys must be sorted in alphabetical order"
            )
        for test_name, test_option in test_to_option.items():
            if test_option not in TEST_OPTIONS:
                raise RuntimeError(
                    f"In failures_dict, got value={test_option} but it needs to be in {TEST_OPTIONS}"
                )
            if not any(test_name.startswith(test) for test in test_utils):
                raise RuntimeError(
                    f"In failures_dict, test name '{test_name}' should begin with one of {test_utils}"
                )
            for test in test_utils:
                if not test_name.startswith(test):
                    continue
                base_test_name = test_name[len(test) + 2 :]
                if hasattr(testcase, base_test_name):
                    continue
                raise RuntimeError(
                    f"In failures dict, got test name '{test_name}'. We parsed this as "
                    f"running test '{test}' on '{base_test_name}', but "
                    f"{base_test_name} does not exist on the TestCase. "
                    f"Maybe you need to change the test name?"
                )


class OpCheckMode(TorchFunctionMode):
    """
    For a given test, OpCheckMode intercepts calls to operators and runs
    test_util(op, args, kwargs) for each intercepted (op, args, kwargs).
    """

    def __init__(self, namespaces, test_util, failures_dict, test_name):
        # We will intercept calls to ops with these namespaces
        self.namespaces = namespaces
        # The test utility function. Its signature should be (op, args, kwargs) -> None.
        # Examples of test utilities are: schema_check, make_fx_check
        self.test_util = test_util
        # The name of the test that is running this OpCheckMode.
        self.test_name = test_name
        # Maps qualname -> test_name -> skip/xfail
        # Tells us if we should skip a test or assert that there is a failure.
        self.failures_dict = failures_dict

        # OpCheckMode surpresses errors, collects them here, and then raises them on exit.
        # Maps qualname -> List[exception]
        self.seen_ops_to_errors = {}

    def maybe_raise_errors_on_exit(self):
        # Check expected failures first
        for qualname in self.seen_ops_to_errors.keys():
            option = retrieve(self.failures_dict, qualname, self.test_name)
            if len(self.seen_ops_to_errors[qualname]) == 0:
                if option == "xfail":
                    raise OpCheckError(
                        f"Unexpected success for operator {qualname} on test {self.test_name}"
                    )
                continue
        for qualname in self.seen_ops_to_errors.keys():
            option = retrieve(self.failures_dict, qualname, self.test_name)
            if option != "success":
                continue
            if len(self.seen_ops_to_errors[qualname]) == 0:
                continue
            # Raise the first error
            ex = self.seen_ops_to_errors[qualname][0]
            raise OpCheckError(
                f"{self.test_name} failed on operator {qualname}"
            ) from ex

    def __exit__(self, *args, **kwargs):
        try:
            self.maybe_raise_errors_on_exit()
        finally:
            result = super().__exit__(*args, **kwargs)
        return result

    def run_test_util(self, op, args, kwargs):
        try:
            self.test_util(op, args, kwargs)
        except torch._subclasses.fake_tensor.UnsupportedFakeTensorException:
            # We might get here if the input is already a FakeTensor
            # or if we're in a torch.compile block. Just ignore these
            # since we can't handle them and reporting them as failures
            # is too noisy.
            pass

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        # Only intercept calls to operators
        if not isinstance(func, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
            return func(*args, **kwargs)
        # Pre-existing code may not use the .default overload. If we see an
        # OpOverloadPacket and we cannot resolve the overload, then we just throw
        # and ask the user to clarify. Otherwise, we attempt to resolve the overload.
        if isinstance(func, torch._ops.OpOverloadPacket):
            func = resolve_unique_overload_or_throw(func)
        qualname = func.name()
        ns = qualname.split("::")[0]
        if ns not in self.namespaces:
            return func(*args, **kwargs)

        args_c, kwargs_c = pytree.tree_map(clone_input, (args, kwargs))
        # Only call test_util(op, *args, **kwargs) if this succeeds.
        result = func(*args, **kwargs)

        option = retrieve(self.failures_dict, qualname, self.test_name)
        if option == "success" or option == "xfail":
            # Surpress all errors during execution. Raise them during __exit__.
            try:
                if qualname not in self.seen_ops_to_errors:
                    self.seen_ops_to_errors[qualname] = []
                self.run_test_util(func, args_c, kwargs_c)
            except Exception as ex:
                self.seen_ops_to_errors[qualname].append(ex)
        elif option == "skip":
            pass
        return result


class OpCheckError(Exception):
    pass


def resolve_unique_overload_or_throw(op: torch._ops.OpOverloadPacket):
    all_schemas = torch._C._jit_get_schemas_for_operator(op._qualified_op_name)
    if len(all_schemas) != 1:
        raise RuntimeError(
            f"opcheck can only test operators without overloads. "
            f"Got the following overloads for {op._qualified_op_name}: "
            f"{[schema.overload_name for schema in all_schemas]}"
        )

    overload_name = all_schemas[0].overload_name
    if overload_name == "":
        return op.default
    return getattr(op, overload_name)


def retrieve(failures_dict, qualname, test_name):
    if qualname not in failures_dict:
        return "success"
    dct = failures_dict[qualname]
    if test_name not in dct:
        return "success"
    return dct[test_name]
