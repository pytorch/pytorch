# Owner(s): ["module: pallas"]

"""Tests for torch.library.pallas_op.

Tests that do not require JAX or TPU hardware test the signature
introspection, dtype mapping, and schema inference utilities.
Tests that require JAX are skipped when JAX is not installed.
"""

import inspect
import types
import unittest

import torch
from torch._library.pallas import (
    _get_torch_signature,
    _infer_static_argnums,
    _verify_signature,
    pallas_op,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _jax_available():
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


class SignatureVerificationTest(TestCase):
    """Tests for signature verification without requiring JAX."""

    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_valid_signature(self):
        import jax

        def fn(x: jax.Array, y: jax.Array) -> jax.Array:
            pass

        _verify_signature(inspect.signature(fn))

    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_valid_signature_with_static_args(self):
        import jax

        def fn(x: jax.Array, n: int, scale: float) -> jax.Array:
            pass

        _verify_signature(inspect.signature(fn))

    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_valid_signature_optional_tensor(self):
        import jax

        def fn(x: jax.Array, bias: jax.Array | None) -> jax.Array:
            pass

        _verify_signature(inspect.signature(fn))

    def test_missing_annotation_raises(self):
        def fn(x, y):
            pass

        with self.assertRaises(ValueError, msg="Missing argument type annotation"):
            _verify_signature(inspect.signature(fn))

    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_invalid_arg_type_raises(self):
        import jax

        def fn(x: jax.Array, y: list) -> jax.Array:
            pass

        with self.assertRaises(ValueError, msg="invalid"):
            _verify_signature(inspect.signature(fn))


class StaticArgnumInferenceTest(TestCase):
    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_all_tensors(self):
        import jax

        def fn(x: jax.Array, y: jax.Array) -> jax.Array:
            pass

        result = _infer_static_argnums(inspect.signature(fn))
        self.assertEqual(result, ())

    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_mixed_args(self):
        import jax

        def fn(x: jax.Array, n: int, y: jax.Array, scale: float) -> jax.Array:
            pass

        result = _infer_static_argnums(inspect.signature(fn))
        self.assertEqual(result, (1, 3))

    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_optional_tensor_not_static(self):
        import jax

        def fn(x: jax.Array, bias: jax.Array | None) -> jax.Array:
            pass

        result = _infer_static_argnums(inspect.signature(fn))
        self.assertEqual(result, ())


class TorchSignatureConversionTest(TestCase):
    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_basic_conversion(self):
        import jax

        def fn(x: jax.Array, y: jax.Array) -> jax.Array:
            pass

        torch_sig = _get_torch_signature(inspect.signature(fn))
        params = list(torch_sig.parameters.values())
        self.assertEqual(params[0].annotation, torch.Tensor)
        self.assertEqual(params[1].annotation, torch.Tensor)
        self.assertEqual(torch_sig.return_annotation, torch.Tensor)

    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_mixed_types_conversion(self):
        import jax

        def fn(x: jax.Array, n: int, scale: float) -> jax.Array:
            pass

        torch_sig = _get_torch_signature(inspect.signature(fn))
        params = list(torch_sig.parameters.values())
        self.assertEqual(params[0].annotation, torch.Tensor)
        self.assertEqual(params[1].annotation, int)
        self.assertEqual(params[2].annotation, float)

    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_optional_conversion(self):
        import jax

        def fn(x: jax.Array, bias: jax.Array | None) -> jax.Array:
            pass

        torch_sig = _get_torch_signature(inspect.signature(fn))
        params = list(torch_sig.parameters.values())
        self.assertEqual(params[1].annotation, torch.Tensor | types.NoneType)

    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_tuple_return(self):
        import jax

        def fn(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            pass

        torch_sig = _get_torch_signature(inspect.signature(fn))
        self.assertEqual(torch_sig.return_annotation, tuple[torch.Tensor, torch.Tensor])


class PallasOpRegistrationTest(TestCase):
    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError, msg="namespace::name"):

            @pallas_op("no_namespace")
            def fn(x: torch.Tensor) -> torch.Tensor:
                return x

    def test_double_colon_required(self):
        with self.assertRaises(ValueError, msg="namespace::name"):

            @pallas_op("a::b::c")
            def fn(x: torch.Tensor) -> torch.Tensor:
                return x


class PallasOpDiscoverabilityTest(TestCase):
    def test_available_in_torch_library(self):
        self.assertTrue(hasattr(torch.library, "pallas_op"))

    def test_in_all(self):
        import torch.library as lib

        self.assertIn("pallas_op", lib.__all__)


class DtypeMappingTest(TestCase):
    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_roundtrip_common_dtypes(self):
        from torch._library.pallas import _jax_to_torch_dtype, _torch_to_jax_dtype

        for dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]:
            jax_dtype = _torch_to_jax_dtype(dtype)
            roundtrip = _jax_to_torch_dtype(jax_dtype)
            self.assertEqual(dtype, roundtrip)

    @unittest.skipUnless(_jax_available(), "JAX not installed")
    def test_unsupported_dtype_raises(self):
        from torch._library.pallas import _torch_to_jax_dtype

        with self.assertRaises(NotImplementedError):
            _torch_to_jax_dtype(torch.qint8)


if __name__ == "__main__":
    run_tests()
