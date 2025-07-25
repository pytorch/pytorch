# Owner(s): ["module: codegen"]

from __future__ import annotations

import tempfile
import unittest

import expecttest

from torchgen.gen import _GLOBAL_PARSE_NATIVE_YAML_CACHE  # noqa: F401
from torchgen.gen_backend_stubs import run


# gen_backend_stubs.py is an integration point that is called directly by external backends.
# The tests here are to confirm that badly formed inputs result in reasonable error messages.
class TestGenBackendStubs(expecttest.TestCase):
    def setUp(self) -> None:
        _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()

    def assert_success_from_gen_backend_stubs(self, yaml_str: str) -> None:
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(yaml_str)
            fp.flush()
            run(fp.name, "", True)

    def get_errors_from_gen_backend_stubs(
        self, yaml_str: str, *, kernels_str: str | None = None
    ) -> str:
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(yaml_str)
            fp.flush()
            try:
                if kernels_str is None:
                    run(fp.name, "", True)
                else:
                    with tempfile.NamedTemporaryFile(mode="w") as kernel_file:
                        kernel_file.write(kernels_str)
                        kernel_file.flush()
                        run(fp.name, "", True, impl_path=kernel_file.name)
            except AssertionError as e:
                # Scrub out the temp file name from any error messages to simplify assertions.
                return str(e).replace(fp.name, "")
            self.fail(
                "Expected gen_backend_stubs to raise an AssertionError, but it did not."
            )

    def test_valid_single_op(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_multiple_ops(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- add.Tensor
- abs"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_zero_ops(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_zero_ops_doesnt_require_backend_dispatch_key(self) -> None:
        yaml_str = """\
backend: BAD_XLA
cpp_namespace: torch_xla
supported:"""
        # External codegen on a yaml file with no operators is effectively a no-op,
        # so there's no reason to parse the backend
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_with_autograd_ops(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs
autograd:
- add.Tensor"""
        # External codegen on a yaml file with no operators is effectively a no-op,
        # so there's no reason to parse the backend
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_missing_backend(self) -> None:
        yaml_str = """\
cpp_namespace: torch_xla
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error, '''You must provide a value for "backend"'''
        )

    def test_empty_backend(self) -> None:
        yaml_str = """\
backend:
cpp_namespace: torch_xla
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error, '''You must provide a value for "backend"'''
        )

    def test_backend_invalid_dispatch_key(self) -> None:
        yaml_str = """\
backend: NOT_XLA
cpp_namespace: torch_xla
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """\
unknown dispatch key NOT_XLA
  The provided value for "backend" must be a valid DispatchKey, but got NOT_XLA.""",
        )  # noqa: B950

    def test_missing_cpp_namespace(self) -> None:
        yaml_str = """\
backend: XLA
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error, '''You must provide a value for "cpp_namespace"'''
        )

    def test_whitespace_cpp_namespace(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace:\t
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error, '''You must provide a value for "cpp_namespace"'''
        )

    # supported is a single item (it should be a list)
    def test_nonlist_supported(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported: abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """expected "supported" to be a list, but got: abs (of type <class 'str'>)""",
        )

    # supported contains an op that isn't in native_functions.yaml
    def test_supported_invalid_op(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs_BAD"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error, """Found an invalid operator name: abs_BAD"""
        )

    # The backend is valid, but doesn't have a valid autograd key. They can't override autograd kernels in that case.
    # Only using Vulkan here because it has a valid backend key but not an autograd key- if this changes we can update the test.
    def test_backend_has_no_autograd_key_but_provides_entries(self) -> None:
        yaml_str = """\
backend: Vulkan
cpp_namespace: torch_vulkan
supported:
- add
autograd:
- sub"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error, """Found an invalid operator name: add"""
        )  # noqa: B950

    # in an operator group, currently all operators must either be registered to the backend or autograd kernel.
    # Here, functional and out mismatch
    def test_backend_autograd_kernel_mismatch_out_functional(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- add.Tensor
autograd:
- add.out"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add_out is listed under "autograd".""",  # noqa: B950
        )

    # in an operator group, currently all operators must either be registered to the backend or autograd kernel.
    # Here, functional and inplace mismatch
    def test_backend_autograd_kernel_mismatch_functional_inplace(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- add.Tensor
autograd:
- add_.Tensor"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add_ is listed under "autograd".""",  # noqa: B950
        )

    # Currently, the same operator can't be listed under both 'supported' and 'autograd', which would
    # involve registering the same kernel to both the XLA and AutogradXLA keys.
    # If we need that functionality in the future, we'll need to augment the codegen.
    def test_op_appears_in_supported_and_autograd_lists(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- add.Tensor
autograd:
- add.Tensor"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add is listed under "autograd".""",  # noqa: B950
        )

    # unrecognized extra yaml key
    def test_unrecognized_key(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs
invalid_key: invalid_val"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """ contains unexpected keys: invalid_key. Only the following keys are supported: backend, class_name, cpp_namespace, extra_headers, supported, autograd, full_codegen, non_native, ir_gen, symint""",  # noqa: B950
        )

    # if use_out_as_primary is provided, it must be a bool
    def test_use_out_as_primary_non_bool(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
use_out_as_primary: frue
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """You must provide either True or False for use_out_as_primary. Provided: frue""",
        )  # noqa: B950

    # if device_guard is provided, it must be a bool
    def test_device_guard_non_bool(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
device_guard: frue
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """You must provide either True or False for device_guard. Provided: frue""",
        )  # noqa: B950

    def test_incorrect_kernel_name(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs
autograd:
- add.Tensor"""
        # Codegen will expect two kernel names (and try to parse them with regex):
        # XLANativeFunctions::abs(...)
        # XLANativeFunctions::add(...)
        kernels_str = """\
at::Tensor& XLANativeFunctions::absWRONG(at::Tensor& self) {}
at::Tensor& XLANativeFunctions::add(at::Tensor& self) {}"""
        output_error = self.get_errors_from_gen_backend_stubs(
            yaml_str, kernels_str=kernels_str
        )
        self.assertExpectedInline(
            output_error,
            """\

XLANativeFunctions is missing a kernel definition for abs. We found 0 kernel(s) with that name,
but expected 1 kernel(s). The expected function schemas for the missing operator are:
at::Tensor abs(const at::Tensor & self)

""",
        )


if __name__ == "__main__":
    unittest.main()
