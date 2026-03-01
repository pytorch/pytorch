# Owner(s): ["module: fx"]

import inspect
import unittest

import torch.fx.operator_schemas as op_schemas


class TestFastBind(unittest.TestCase):
    def _assert_fast_bind_matches_sig_bind(self, sig, args, kwargs):
        # Success path: compare BoundArguments
        try:
            ref = sig.bind(*args, **kwargs)
        except TypeError:
            with self.assertRaises(TypeError):
                op_schemas._fast_bind(sig, args, kwargs)
            return

        got = op_schemas._fast_bind(sig, args, kwargs)
        self.assertEqual(ref.arguments, got.arguments)
        self.assertEqual(ref.args, got.args)
        self.assertEqual(ref.kwargs, got.kwargs)

        # Also validate default population matches
        ref = sig.bind(*args, **kwargs)
        ref.apply_defaults()
        got = op_schemas._fast_bind(sig, args, kwargs)
        got.apply_defaults()
        self.assertEqual(ref.arguments, got.arguments)
        self.assertEqual(ref.args, got.args)
        self.assertEqual(ref.kwargs, got.kwargs)

    def test_positional_or_keyword_and_defaults(self):
        def f(a, b=1, c=2):
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (10,), {"c": 3})
        self._assert_fast_bind_matches_sig_bind(sig, (10, 20), {})
        self._assert_fast_bind_matches_sig_bind(sig, (), {"a": 1})

    def test_missing_required(self):
        def f(a, b):
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (), {})
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {})

    def test_too_many_positional(self):
        def f(a, b=1):
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (1, 2, 3), {})

    def test_multiple_values_for_argument(self):
        def f(a, b):
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {"a": 2, "b": 3})

    def test_unexpected_keyword(self):
        def f(a, b=1):
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {"c": 3})

    def test_keyword_only(self):
        def f(a, *, b, c=1):
            pass

        sig = inspect.signature(f)
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {"b": 2})
        self._assert_fast_bind_matches_sig_bind(sig, (1, 2), {})
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {})

    def test_positional_only(self):
        # def f(x, /, y, *, z=1): ...
        sig = inspect.Signature(
            [
                inspect.Parameter("x", inspect.Parameter.POSITIONAL_ONLY),
                inspect.Parameter("y", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("z", inspect.Parameter.KEYWORD_ONLY, default=1),
            ]
        )
        self._assert_fast_bind_matches_sig_bind(sig, (1, 2), {"z": 3})
        self._assert_fast_bind_matches_sig_bind(sig, (), {"x": 1, "y": 2})
        self._assert_fast_bind_matches_sig_bind(sig, (1,), {"x": 2, "y": 3})

    def test_from_keyword_positional_only_pattern(self):
        # Mirrors a common TorchScript schema normalization pattern in operator_schemas.py
        # where "from" is treated as positional-only.
        sig = inspect.Signature(
            [
                inspect.Parameter("input", inspect.Parameter.POSITIONAL_ONLY),
                inspect.Parameter("from", inspect.Parameter.POSITIONAL_ONLY, default=0.0),
                inspect.Parameter("to", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1.0),
                inspect.Parameter(
                    "generator",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                ),
            ]
        )
        self._assert_fast_bind_matches_sig_bind(sig, ("t",), {})
        self._assert_fast_bind_matches_sig_bind(sig, ("t", 0.25), {})
        self._assert_fast_bind_matches_sig_bind(sig, ("t", 0.25, 0.75), {"generator": None})
        self._assert_fast_bind_matches_sig_bind(sig, (), {"input": "t", "from": 0.1})

    def test_varargs_and_varkw_fallback(self):
        def f(a, *args, b=0, **kwargs):
            pass

        sig = inspect.signature(f)
        # Extra positional should be captured by *args; extra keywords by **kwargs
        self._assert_fast_bind_matches_sig_bind(sig, (1, 2, 3), {"b": 4, "x": 5})
        self._assert_fast_bind_matches_sig_bind(sig, (), {"a": 1, "x": 2})


if __name__ == "__main__":
    unittest.main()

