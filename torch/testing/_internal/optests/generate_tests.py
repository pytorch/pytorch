import datetime
import difflib
import functools
import json
import os
import tempfile
import unittest

import torch

import torch._dynamo

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
    args, kwargs = deepcopy_tensors((args, kwargs))
    with SchemaCheckMode():
        result = op(*args, **kwargs)
        return result


def safe_autograd_registration_check(op, args, kwargs):
    # Don't perform autograd_registration_check if none of the inputs require grad.
    if not pytree.tree_any_only(
        torch.Tensor, lambda x: x.requires_grad, (args, kwargs)
    ):
        return
    args, kwargs = deepcopy_tensors((args, kwargs))
    return autograd_registration_check(op, args, kwargs)


def safe_fake_check(op, args, kwargs):
    args, kwargs = deepcopy_tensors((args, kwargs))
    return fake_check(op, args, kwargs, dynamic_only=False)


def safe_aot_autograd_check(op, args, kwargs, dynamic):
    def func(*args, **kwargs):
        args, kwargs = pytree.tree_map_only(torch.Tensor, torch.clone, (args, kwargs))
        return op(*args, **kwargs)

    # aot_autograd_check runs func(*args, **kwargs) multiple times
    # and assumes `func` does not modify its inputs.
    return aot_autograd_check(func, args, kwargs, dynamic, check_gradients="auto")


def deepcopy_tensors(inputs):
    return pytree.tree_map_only(torch.Tensor, clone_input, inputs)


# Test util requirements
# - The test util must have signature (op: OpOverload, args, kwargs)
# - The test util must NOT mutate args, kwargs.
# - The test utils in this list must not be prefixes of each other. For example,
#   having both "test_schema" and "test_schema_is_functional" is NOT OK.
# - The order of items in this dict matters (for opcheck), we'll run them
#   in order.
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

GDOC = "https://docs.google.com/document/d/1Pj5HRZvdOq3xpFpbEjUZp2hBovhy7Wnxw14m6lF2154/edit"


def generate_opcheck_tests(
    testcase,
    namespaces,
    failures_dict_path,
    additional_decorators,
    test_utils,
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

    For more details, see https://docs.google.com/document/d/1Pj5HRZvdOq3xpFpbEjUZp2hBovhy7Wnxw14m6lF2154/edit

    Args:
        testcase: The testcase we will modify and generate additional tests for.
        namespaces: We will only intercept calls to custom operators with these
                    namespaces.
        failures_dict_path: See ``validate_failures_dict_structure`` for more details
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
    failures_dict = FailuresDict.load(
        failures_dict_path, create_file=should_update_failures_dict()
    )
    validate_failures_dict_structure(failures_dict, test_utils, testcase)
    validate_failures_dict_formatting(failures_dict_path)

    def construct_method(attr, prefix, tester):
        method = getattr(testcase, attr)
        new_method_name = prefix + "__" + attr

        def new_method(*args, **kwargs):
            with OpCheckMode(
                namespaces,
                prefix,
                tester,
                failures_dict,
                new_method_name,
                failures_dict_path,
            ):
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


def validate_failures_dict_formatting(failures_dict_path):
    with open(failures_dict_path) as fp:
        actual = fp.read()
    failures_dict = FailuresDict.load(failures_dict_path)
    expected = failures_dict._save(to_str=True)
    if actual == expected:
        return
    if should_update_failures_dict():
        failures_dict = FailuresDict.load(failures_dict_path)
        failures_dict.save()
        return
    expected = expected.splitlines(1)
    actual = actual.splitlines(1)
    diff = difflib.unified_diff(expected, actual)
    diff = "".join(diff)
    raise RuntimeError(
        f"\n{diff}\n\nExpected the failures dict to be formatted "
        f"a certain way. Please see the above diff; you can correct "
        f"this either manually or by re-running the test with "
        f"PYTORCH_OPCHECK_ACCEPT=1"
    )


def validate_failures_dict_structure(failure_dict, test_utils, testcase):
    """Validates the failures dict.

    The failure dict looks something like the following.
    It maps operator name (qualname) to a list of autogenerated tests.
    Each autogenerated test may have a check for the operator (if the operator is
    called by the test); the dictionary specifies if we should skip the check,
    or if we expect some check to fail.

    {
        "fbgemm::split_lengths": {
            "test_schema__test_split_lengths": {
                "comment": "you can put whatever you want into the comment section",
                "status": "xfail",
            }
            "test_schema__test_split_lengths_empty": {
                "comment": "",
                "status": "skip",
            },
        },
        "fbgemm::gather_lengths": {
            "test_schema__test_gather_lengths": {
                "comment": "",
                "status": "skip",
            },
        },
    }

    """
    failure_dict = failure_dict.data
    qualnames = list(failure_dict.keys())
    for test_to_option in failure_dict.values():
        test_names = list(test_to_option.keys())
        for test_name, test_dict in test_to_option.items():
            if set(test_dict.keys()) != set({"comment", "status"}):
                raise RuntimeError(
                    "in failures_dict, expected sub-dict to have keys 'comment' and 'status'"
                )
            test_option = test_dict["status"]
            if test_option not in TEST_OPTIONS:
                raise RuntimeError(
                    f"In failures_dict, got status={test_option} but it needs to be in {TEST_OPTIONS}"
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


def should_update_failures_dict():
    key = "PYTORCH_OPCHECK_ACCEPT"
    return key in os.environ and os.environ[key] == "1"


class OpCheckMode(TorchFunctionMode):
    """
    For a given test, OpCheckMode intercepts calls to operators and runs
    test_util(op, args, kwargs) for each intercepted (op, args, kwargs).
    """

    def __init__(
        self,
        namespaces,
        test_util_name,
        test_util,
        failures_dict,
        test_name,
        failures_dict_path,
    ):
        # We will intercept calls to ops with these namespaces
        self.namespaces = namespaces
        # The test utility function. Its signature should be (op, args, kwargs) -> None.
        # Examples of test utilities are: schema_check, make_fx_check
        self.test_util = test_util
        self.test_util_name = test_util_name
        # The name of the test that is running this OpCheckMode.
        self.test_name = test_name
        # Maps qualname -> test_name -> skip/xfail
        # Tells us if we should skip a test or assert that there is a failure.
        self.failures_dict = failures_dict
        # Location of the failures dict. Makes it so that the error message is better.
        self.failures_dict_path = failures_dict_path

        # OpCheckMode surpresses errors, collects them here, and then raises them on exit.
        # Maps qualname -> List[exception]
        self.seen_ops_to_errors = {}

    def maybe_raise_errors_on_exit(self):
        # Check expected failures first
        for qualname in self.seen_ops_to_errors.keys():
            option = self.failures_dict.get_status(qualname, self.test_name)
            if len(self.seen_ops_to_errors[qualname]) == 0:
                if should_update_failures_dict():
                    self.failures_dict.set_status(
                        qualname, self.test_name, "success", comment=""
                    )
                else:
                    if option == "xfail":
                        raise OpCheckError(
                            f"generate_opcheck_tests: Unexpected success for operator "
                            f"{qualname} on test {self.test_name}. This may mean that "
                            f"you have fixed this test failure. Please rerun the test with "
                            f"PYTORCH_OPCHECK_ACCEPT=1 to automatically update the test runner "
                            f"or manually remove the "
                            f"expected failure in the failure dict at "
                            f"{self.failures_dict_path}"
                            f"For more details, see "
                            f"{GDOC}"
                        )
                continue
        failed_ops = []
        for qualname in self.seen_ops_to_errors.keys():
            option = self.failures_dict.get_status(qualname, self.test_name)
            if option != "success":
                continue
            if len(self.seen_ops_to_errors[qualname]) == 0:
                continue
            failed_ops.append(qualname)
        if not failed_ops:
            return

        if should_update_failures_dict():
            for op in failed_ops:
                self.failures_dict.set_status(op, self.test_name, "xfail")
            return

        # Raise from the first error but also report about all of them to make
        # recording xfails easier.
        ex, op, args, kwargs = self.seen_ops_to_errors[failed_ops[0]][0]
        if should_print_repro():
            repro_command = generate_repro(self.test_util_name, op, args, kwargs)
            repro_command = (
                f"\n\nFor a minimal repro, run the following: \n\n{repro_command}"
            )
        else:
            repro_command = ""
        raise OpCheckError(
            f"Test generated by `generate_opcheck_tests`, {self.test_name}, "
            f"failed on operators {failed_ops}. This usually means that the "
            f"operators are not implemented correctly and may lead to silently "
            f"incorrect behavior. Set PYTORCH_OPCHECK_PRINT_REPRO=1 for a standalone repro, "
            f"or please see "
            f"{GDOC} "
            f"for more recommendations. "
            f"{repro_command}"
        ) from ex

    def __exit__(self, *args, **kwargs):
        try:
            self.maybe_raise_errors_on_exit()
            if should_update_failures_dict():
                self.failures_dict.save()
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
        if (
            torch.jit.is_tracing()
            or torch.jit.is_scripting()
            or torch._dynamo.is_compiling()
        ):
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

        args_c, kwargs_c = deepcopy_tensors((args, kwargs))
        # Only call test_util(op, *args, **kwargs) if this succeeds.
        result = func(*args, **kwargs)

        option = self.failures_dict.get_status(qualname, self.test_name)
        if option == "success" or option == "xfail":
            # Surpress all errors during execution. Raise them during __exit__.
            try:
                if qualname not in self.seen_ops_to_errors:
                    self.seen_ops_to_errors[qualname] = []
                self.run_test_util(func, args_c, kwargs_c)
            except Exception as ex:
                if should_print_repro():
                    self.seen_ops_to_errors[qualname].append((ex, func, args, kwargs))
                else:
                    self.seen_ops_to_errors[qualname].append((ex, None, None, None))
        elif option == "skip":
            pass
        return result


def should_print_repro():
    """If set, the tests generated by `generate_opcheck_tests` will print a
    repro command on failure.

    In order to print the repro command, we need to save some tensors to disk.
    These will be saved under the following directory:
    {tempfile.gettempdir()}/pytorch_opcheck_safe_to_delete/.

    Although this is a temp folder, it will usually not automatically get cleaned
    up, so you'll need to manually delete it.
    """
    key = "PYTORCH_OPCHECK_PRINT_REPRO"
    if key not in os.environ:
        return False
    value = os.environ[key]
    return value == "1" or value == 1


def opcheck(op, args, kwargs=None, *, test_utils="ALL", raise_exception=True):
    """Given an operator and some sample arguments, tests if the operator is
    registered correctly.

    We test the following (which are important for correctness in eager-mode
    PyTorch and with torch.compile):
    - test_schema: if the operator's schema is correct.
    - test_autograd_registration: if autograd was registered correctly,
        i.e. to the correct DispatchKey.
    - test_faketensor: If the operator has a FakeTensor implementation
        (and if it is correct).
    - test_aot_dispatch_static: If the operator works with
        AOTAutograd/AOTDispatch, which is one of the parts in the PT2 stack.
        Checks that the outputs (and gradients, if they are computable)
        of the operator are the same under eager-mode PyTorch and torch.compile.
    - test_aot_dispatch_dynamic: Same as aot_dispatch_static, but
        tests dynamic shapes instead of static shapes.

    For best results, please call ``opcheck`` multiple times with a
    representative set of inputs. For example, if your operator supports
    autograd, please use ``opcheck`` with inputs that require_grad.

    Args:
        op: The operator. Should look like torch.ops.aten.foo
        args: The args to the operator
        kwargs: The kwargs to the operator
        test_utils: Tests that we should run. Default: all of them.
            Example: ["test_schema", "test_faketensor"]
        raise_exception: If we should raise an exception on the first
            error. If False, we will return a dict with information
            on if each test passed or not.

    """

    if kwargs is None:
        kwargs = {}
    if isinstance(op, torch._ops.OpOverloadPacket):
        op = resolve_unique_overload_or_throw(op)
    if not isinstance(op, torch._ops.OpOverload):
        raise ValueError(
            f"opcheck(op, ...): op must be instance of torch._ops.OpOverload, "
            f"e.g. torch.ops.aten.sin.default, got {type(op)}"
        )
    if test_utils == "ALL":
        test_utils = tuple(ALL_TEST_UTILS.keys())
    if isinstance(test_utils, str):
        test_utils = (test_utils,)
    if not isinstance(test_utils, (tuple, list)) or not set(test_utils).issubset(
        ALL_TEST_UTILS.keys()
    ):
        raise ValueError(
            f"opcheck(op, ..., test_utils={test_utils}), expected test_utils "
            f"to be subset of {tuple(ALL_TEST_UTILS.keys())} but it was not"
        )

    results_dict = {}
    for test_util in test_utils:
        tester = ALL_TEST_UTILS[test_util]
        try:
            tester(op, args, kwargs)
            results_dict[test_util] = "SUCCESS"
        except Exception as ex:
            if raise_exception:
                raise OpCheckError(
                    f"opcheck(op, ...): {test_util} failed with {ex} "
                    f"(scroll up for stack trace)"
                ) from ex
            results_dict[test_util] = ex
    return results_dict


class OpCheckError(Exception):
    pass


def generate_repro(test, op, args, kwargs):
    now = datetime.datetime.now()
    unix_timestamp = datetime.datetime.timestamp(now) * 1000
    path = os.path.join(tempfile.gettempdir(), "pytorch_opcheck_safe_to_delete")
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, f"repro_{unix_timestamp}.pt")

    ns, name = op._schema.name.split("::")
    overload = op._overloadname

    repro_command = (
        f"import torch\n"
        f"from torch.testing._internal.optests import opcheck\n"
        f"# Make sure you have loaded the library that contains the op\n"
        f"# via an import or torch.ops.load_library(...)\n"
        f"op = torch.ops.{ns}.{name}.{overload}\n"
        f'args, kwargs = torch.load("{filepath}")\n'
        f'opcheck(op, args, kwargs, test_utils="{test}")\n'
    )
    torch.save((args, kwargs), filepath)
    return repro_command


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


DUMP_OPTIONS = {"indent": 2, "sort_keys": True}


class FailuresDict:
    def __init__(self, path, data):
        self.path = path
        self.data = data

    @staticmethod
    def load(path, *, create_file=False):
        if create_file and not os.path.exists(path):
            result = FailuresDict(path, {})
            FailuresDict.save()
            return result
        with open(path) as fp:
            dct = json.load(fp)
        assert "data" in dct
        assert "_version" in dct and dct["_version"] == 1
        return FailuresDict(path, dct["data"])

    def _save(self, to_str=False):
        to_dump = {
            "_description": (
                f"This is a dict containing failures for tests autogenerated by "
                f"generate_opcheck_tests. "
                f"For more details, please see {GDOC}"
            ),
            "data": self.data,
            "_version": 1,
        }
        # json.dumps doesn't end with a newline. Let's add one because files
        # should end in newlines.
        serialized = json.dumps(to_dump, **DUMP_OPTIONS) + "\n"
        if to_str:
            return serialized
        with open(self.path, "w") as fp:
            fp.write(serialized)
        return None

    def save(self):
        return self._save()

    def get_status(self, qualname, test_name):
        if qualname not in self.data:
            return "success"
        dct = self.data[qualname]
        if test_name not in dct:
            return "success"
        return dct[test_name]["status"]

    def set_status(self, qualname, test_name, status, *, comment=None):
        if qualname not in self.data:
            self.data[qualname] = {}
        dct = self.data[qualname]
        if test_name not in dct:
            dct[test_name] = {"status": None, "comment": ""}

        if status == "success":
            # The default status is "success".
            del dct[test_name]
        else:
            dct[test_name]["status"] = status
            if comment is not None:
                dct[test_name]["comment"] = comment
