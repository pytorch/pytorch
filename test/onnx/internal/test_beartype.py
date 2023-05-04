# Owner(s): ["module: onnx"]
"""Unit tests for the internal beartype wrapper module."""

import unittest

from torch.onnx._internal import _beartype
from torch.testing._internal import common_utils


def beartype_installed():
    try:
        import beartype  # noqa: F401
    except ImportError:
        return False
    return True


def skip_if_beartype_not_installed(test_case):
    return unittest.skipIf(not beartype_installed(), "beartype is not installed")(
        test_case
    )


def func_with_type_hint(x: int) -> int:
    return x


def func_with_incorrect_type_hint(x: int) -> str:
    return x  # type: ignore[return-value]


@common_utils.instantiate_parametrized_tests
class TestBeartype(common_utils.TestCase):
    def test_create_beartype_decorator_returns_no_op_decorator_when_disabled(self):
        decorator = _beartype._create_beartype_decorator(
            _beartype.RuntimeTypeCheckState.DISABLED,
        )
        decorated = decorator(func_with_incorrect_type_hint)
        decorated("string_input")  # type: ignore[arg-type]

    @skip_if_beartype_not_installed
    def test_create_beartype_decorator_warns_when_warnings(self):
        decorator = _beartype._create_beartype_decorator(
            _beartype.RuntimeTypeCheckState.WARNINGS,
        )
        decorated = decorator(func_with_incorrect_type_hint)
        with self.assertWarns(_beartype.CallHintViolationWarning):
            decorated("string_input")  # type: ignore[arg-type]

    @common_utils.parametrize("arg", [1, "string_input"])
    @skip_if_beartype_not_installed
    def test_create_beartype_decorator_errors_when_errors(self, arg):
        import beartype

        decorator = _beartype._create_beartype_decorator(
            _beartype.RuntimeTypeCheckState.ERRORS,
        )
        decorated = decorator(func_with_incorrect_type_hint)
        with self.assertRaises(beartype.roar.BeartypeCallHintViolation):
            decorated(arg)

    @skip_if_beartype_not_installed
    def test_create_beartype_decorator_warning_calls_function_once(self):
        call_count = 0

        def func_with_incorrect_type_hint_and_side_effect(x: int) -> str:
            nonlocal call_count
            call_count += 1
            return x  # type: ignore[return-value]

        decorator = _beartype._create_beartype_decorator(
            _beartype.RuntimeTypeCheckState.WARNINGS,
        )
        decorated = decorator(func_with_incorrect_type_hint_and_side_effect)
        decorated("string_input")  # type: ignore[arg-type]
        self.assertEqual(call_count, 1)
        decorated(1)
        # The return value violates the type hint, but the function is called
        # only once.
        self.assertEqual(call_count, 2)


if __name__ == "__main__":
    common_utils.run_tests()
