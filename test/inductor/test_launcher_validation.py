# Owner(s): ["module: inductor"]
import types

from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.test_case import run_tests, TestCase


def _make_autotuner():
    """Return a minimal stub sufficient for calling _validate_launcher_args."""
    stub = types.SimpleNamespace(fn=types.SimpleNamespace(__name__="test_kernel"))
    return stub


class TestLauncherValidation(TestCase):
    def test_argument_mismatch_error(self):
        def mock_launcher(a, b, stream):
            pass

        mock_launcher.def_arg_names = ["a", "b"]
        autotuner = _make_autotuner()

        # Too many positional args
        with self.assertRaisesRegex(
            TypeError,
            r"expected at most 2 positional arguments \(a, b\) but got 3",
        ):
            CachingAutotuner._validate_launcher_args(
                autotuner, mock_launcher, [1, 2, 3], {}
            )

        # Too few positional args, missing kwargs
        with self.assertRaisesRegex(
            TypeError,
            r"expected 2 arguments \(a, b\) but only 1 were provided via "
            r"positional and keyword arguments\. Missing arguments: b",
        ):
            CachingAutotuner._validate_launcher_args(
                autotuner, mock_launcher, [1], {}
            )

        # Too few args with unknown kwargs (still missing 'b')
        with self.assertRaisesRegex(
            TypeError,
            r"expected 2 arguments \(a, b\) but only 1 were provided via "
            r"positional and keyword arguments\. Missing arguments: b",
        ):
            CachingAutotuner._validate_launcher_args(
                autotuner, mock_launcher, [1], {"c": 2}
            )

        # Overlapping positional and keyword arguments
        with self.assertRaisesRegex(
            TypeError,
            r"got multiple values for argument\(s\) 'b'\. "
            r"This usually means you passed too many positional arguments that overlapped with keyword arguments\.",  # noqa: B950
        ):
            CachingAutotuner._validate_launcher_args(
                autotuner, mock_launcher, [1, 2], {"b": 3}
            )

    def test_kwargs_arguments_no_error(self):
        def mock_launcher(a, b, stream):
            pass

        mock_launcher.def_arg_names = ["a", "b"]
        autotuner = _make_autotuner()

        # Mixed positional and keyword arguments — no error.
        CachingAutotuner._validate_launcher_args(
            autotuner, mock_launcher, [1], {"b": 2}
        )

        # After first success, _inductor_args_validated is cached —
        # second call skips validation.
        CachingAutotuner._validate_launcher_args(
            autotuner, mock_launcher, [], {"a": 1, "b": 2}
        )

    def test_launcher_none_is_noop(self):
        autotuner = _make_autotuner()
        CachingAutotuner._validate_launcher_args(autotuner, None, [1, 2], {"c": 3})

    def test_launcher_varargs_fallback(self):
        def mock_launcher_varargs(*args, **kwargs):
            pass

        def mock_launcher_no_args():
            pass

        autotuner = _make_autotuner()

        # *args launcher — no error regardless of arg count.
        CachingAutotuner._validate_launcher_args(
            autotuner, mock_launcher_varargs, [1, 2, 3], {"c": 3}
        )

        # No-arg launcher: expected_count = max(0, co_argcount - 1) = 0
        with self.assertRaisesRegex(
            TypeError, r"expected at most 0 positional arguments but got 1"
        ):
            CachingAutotuner._validate_launcher_args(
                autotuner, mock_launcher_no_args, [1], {}
            )


if __name__ == "__main__":
    run_tests()
