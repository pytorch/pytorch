import functools

import torch

import torch.utils._pytree as pytree

from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch.overrides import TorchFunctionMode
from torch.testing._internal.optests import (
    aot_autograd_check,
    autograd_registration_check,
    fake_check,
)


def safe_schema_check(op, args, kwargs):
    args, kwargs = deepcopy_tensors(args, kwargs)
    with SchemaCheckMode():
        result = op(*args, **kwargs)
        return result


def safe_autograd_registration_check(op, args, kwargs):
    # Don't perform autograd_registration_check if none of the inputs require grad.
    if not pytree.tree_any_only(
        torch.Tensor, lambda x: x.requires_grad, (args, kwargs)
    ):
        return
    args, kwargs = deepcopy_tensors(args, kwargs)
    return autograd_registration_check(op, args, kwargs)


def safe_fake_check(op, args, kwargs):
    args, kwargs = deepcopy_tensors(args, kwargs)
    return fake_check(op, args, kwargs, dynamic_only=False)


def safe_aot_autograd_check(op, args, kwargs, dynamic):
    def func(*args, **kwargs):
        args, kwargs = pytree.tree_map_only(torch.Tensor, torch.clone, (args, kwargs))
        return op(*args, **kwargs)

    # aot_autograd_check runs func(*args, **kwargs) multiple times
    # and assumes `func` does not modify its inputs.
    return aot_autograd_check(func, args, kwargs, dynamic, check_gradients="auto")


# Test requirements
# - The tests must have signature (op: OpOverload, args, kwargs)
# - The tests must NOT mutate args, kwargs.
# - The tests in this list must not be prefixes of each other. For example,
#   having both "test_schema" and "test_schema_is_functional" is NOT OK.
ALL_TESTS = {
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
    testcase, namespaces, failures_dict, additional_decorators, tests
):
    """Given an existing TestCase, use the existing tests to generate
    additional validation tests for custom operators.

    For all existing tests in the TestCase, we will generate one new
    test for:
    - test_schema: schema correctness
    - test_autograd_registration: autograd registration
    - test_faketensor: faketensor rule
    - test_aot_dispatch_static:
        AOTDispatch is a component in the PT2 stack
        (Dynamo -> "AOTDispatch/AOTAutograd" -> Inductor) that traces out
        a forwards (and optionally a backwards) graph. This tests that
        component with static shapes.
    - test_aot_dispatch_dynamic: AOTDispatch test with dynamic shapes.

    The generated test will have name ``test_<something>__<original name>``.
    For example, if there is a method named ``test_cumsum``, then
    we will generate a ``test_schema__test_cumsum``,
    ``test_faketensor__test_cumsum``, etc.

    Args:
        testcase: The testcase we will modify and generate additional tests for.
        namespaces: The namespaces of the custom operators we will test during the additional tests.
        failures_dict: See ``validate_failures_dict`` for more details
        additional_decorators: Pass us some decorators
        tests: a list of tests to generate. Example: ["test_schema", "test_faketensor"]
    """
    test_methods = [
        m
        for m in dir(testcase)
        if m.startswith("test_") and callable(getattr(testcase, m))
    ]
    validate_failures_dict(failures_dict, tests, testcase)

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

    tests = {name: ALL_TESTS[name] for name in tests}
    for attr in test_methods:
        for prefix, tester in tests.items():
            construct_method(attr, prefix, tester)


TEST_OPTIONS = ("xfail", "skip", "success")


def validate_failures_dict(failure_dict, tests, testcase):
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
            if not any(test_name.startswith(test) for test in tests):
                raise RuntimeError(
                    f"In failures_dict, test name '{test_name}' should begin with one of {tests}"
                )
            for test in tests:
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
        # Give an name for this test
        self.test_name = test_name
        # Maps qualname -> test_name -> skip/xfail
        # Tells us if we should skip a test or assert that there is a failure.
        self.failures_dict = failures_dict

        # NOTE: [OpCheckMode and expected failures]
        # We mark (operator, test_name) pairs as expected failure. However, a test
        # may have multiple invocations of the operator, of which not all
        # may fail. The semantics of the "expected failure" are if ANY invocation
        # of the operator in a test fails.
        #
        # This is implemented via:
        # - We use the OpCheckMode for the entire test
        # - We record which operators we encounter should have an expected failure
        #   and if any invocation failed
        # - When the test is done, we call .validate_xfails()
        self.expecting_failure_on = set({})
        self.received_failure_on = set({})

    def validate_xfails(self):
        diff = self.expecting_failure_on - self.received_failure_on
        if diff:
            raise AssertionError(f"Unexpected success for {diff}, {self.test_name}")

    def __exit__(self, *args, **kwargs):
        try:
            self.validate_xfails()
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

        option = retrieve(self.failures_dict, qualname, self.test_name)
        if option == "success":
            try:
                self.run_test_util(func, args, kwargs)
            except Exception:
                # Useful for debugging.
                identifier = f'("{qualname}", "{self.test_name}")'
                print(f"FAILED: {identifier}")
                raise
        elif option == "skip":
            pass
        elif option == "xfail":
            self.expecting_failure_on.add(qualname)
            try:
                self.run_test_util(func, args, kwargs)
            except Exception:
                self.received_failure_on.add(qualname)
        return func(*args, **kwargs)


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


def deepcopy_tensors(args, kwargs):
    def deepcopy_tensor(x):
        return x.detach().clone().requires_grad_(x.requires_grad)

    return pytree.tree_map_only(torch.Tensor, deepcopy_tensor, (args, kwargs))


def retrieve(failures_dict, qualname, test_name):
    if qualname not in failures_dict:
        return "success"
    dct = failures_dict[qualname]
    if test_name not in dct:
        return "success"
    return dct[test_name]
