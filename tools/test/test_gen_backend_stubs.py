# Owner(s): ["module: codegen"]

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import expecttest

from torchgen import dest
from torchgen.gen import (
    _GLOBAL_PARSE_NATIVE_YAML_CACHE,
    get_grouped_native_functions,
    parse_native_yaml,
)
from torchgen.gen_backend_stubs import parse_backend_yaml, run
from torchgen.model import NativeFunctionsGroup, OperatorName
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, Target


# gen_backend_stubs.py is an integration point that is called directly by external backends.
# The tests here are to confirm that badly formed inputs result in reasonable error messages.
class TestGenBackendStubs(expecttest.TestCase):
    def setUp(self) -> None:
        global _GLOBAL_PARSE_NATIVE_YAML_CACHE
        _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()

    def assert_success_from_gen_backend_stubs(
        self, yaml_str: str, *, kernels_str: str | None = None
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(yaml_str)
            fp.flush()
            if kernels_str is None:
                run(fp.name, "", True)
            else:
                with tempfile.NamedTemporaryFile(mode="w") as kernel_file:
                    kernel_file.write(kernels_str)
                    kernel_file.flush()
                    run(fp.name, "", True, impl_path=kernel_file.name)

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
        )

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
        )

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
            """Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add_out is listed under "autograd".""",
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
            """Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add_ is listed under "autograd".""",
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
            """Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add is listed under "autograd".""",
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
            """ contains unexpected keys: invalid_key. Only the following keys are supported: backend, class_name, cpp_namespace, extra_headers, supported, autograd, full_codegen, non_native, ir_gen, symint""",
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
        )

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
        )

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

    # The per-op dict form (PrivateUse1 out-of-tree structured/out-as-primary
    # feature) lets an op opt into a structured kernel. Its options
    # (structured/device_guard) nest under the op name ("- mul.out:" then an
    # indented "structured: true"); the flat sibling form is rejected
    # (test_flat_dict_shape_rejected).

    # structured: true reuses the native structured kernel for the op.
    def test_valid_per_op_structured(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: true
supported:
- mul.out:
    structured: true"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    # An op may use the out-as-primary redirection without a structured kernel,
    # and override the backend-level device_guard per op.
    def test_valid_per_op_out_as_primary_only(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: true
supported:
- div.out:
    device_guard: false"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    # Per-op dict entries may be mixed with plain string entries in one list.
    def test_valid_per_op_mixed_with_plain(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: true
supported:
- abs
- mul.out:
    structured: true"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    # The full case study from the PR description: structured ops reusing aten's meta
    # and an out-as-primary-only op with a device_guard override.
    def test_valid_per_op_full_case_study(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: true
supported:
- sub.out:
    structured: true
- mul.out:
    structured: true
- div.out:
    device_guard: false"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    # Structured kernels are only generated for PrivateUse1; other keys fail early instead of
    # at codegen with an opaque dispatch-key assertion.
    def test_structured_rejected_for_non_privateuse1(self) -> None:
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
use_out_as_primary: true
supported:
- mul.out:
    structured: true"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Operator 'mul.out' sets 'structured: True', which is only supported for backend PrivateUse1 (got XLA).""",
        )

    # A structured kernel always guards set_output (as in-tree), so a per-op
    # device_guard: false on it would only be half-honored; reject the combination.
    def test_structured_with_device_guard_false_rejected(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: true
supported:
- mul.out:
    structured: true
    device_guard: false"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Operator 'mul.out' sets 'structured: True' with 'device_guard: False'. A structured kernel always guards set_output to place its outputs on the input's device, as in-tree; remove device_guard for this op.""",
        )

    # A quoted YAML scalar parses to a truthy str (not a bool), which would silently be treated
    # as structured; per-op option values must be real booleans.
    def test_per_op_option_non_bool(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
supported:
- mul.out:
    structured: 'false'"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Operator 'mul.out' option 'structured' must be a boolean (true/false), but got str: 'false'.""",
        )

    # The per-op value must be an option mapping, not a list (a common mis-indentation).
    def test_per_op_options_non_dict(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
supported:
- mul.out:
  - structured"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Operator 'mul.out' options must be a mapping of option keys to booleans, but got list: ['structured'].""",
        )

    # A per-op device_guard can only narrow the backend-level flag (every emit site ANDs them),
    # so device_guard: true under a false backend flag would be silently dropped; reject it.
    def test_per_op_device_guard_enable_under_false_backend(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: false
supported:
- div.out:
    device_guard: true"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Operator 'div.out' sets 'device_guard: True' but the backend sets 'device_guard: False'. A per-op device_guard can only disable the backend-level guard, not enable it; set the backend-level device_guard: True and disable it per-op where unwanted.""",
        )

    # A structured kernel is defined as Class::structured_<kernel>::impl(...), or through the
    # TORCH_PRIVATEUSE1_IMPL_FUNC / TORCH_PRIVATEUSE1_FUNC helper macros; the missing-kernel
    # scan must attribute each to <kernel> rather than report it missing.
    def test_missing_kernels_counts_structured_impl(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
supported:
- maximum.out:
    structured: true
- minimum.out:
    structured: true
- div.out"""
        impl = (
            "void PrivateUse1NativeFunctions::structured_maximum_out::impl("
            "const at::Tensor& a, const at::Tensor& b, const at::Tensor& out) {}\n"
            "TORCH_PRIVATEUSE1_IMPL_FUNC(minimum_out)"
            "(const at::Tensor& a, const at::Tensor& b, const at::Tensor& out) {}\n"
            "at::Tensor& TORCH_PRIVATEUSE1_FUNC(div_out)"
            "(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) { return out; }"
        )
        self.assert_success_from_gen_backend_stubs(yaml_str, kernels_str=impl)
        _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str, kernels_str="")
        for kernel in ("maximum_out", "minimum_out", "div_out"):
            self.assertIn(f"missing a kernel definition for {kernel}", output_error)

    # A factory op with no in-tree functional resolves dtype/layout/device in a body aten
    # does not provide, and its .out schema has no options argument, so it is refused.
    # (arange, full, zeros_like have composite functionals and defer to them instead.)
    def test_factory_out_as_primary_rejected(self) -> None:
        for out in ("empty_strided.out", "tril_indices.out"):
            with self.subTest(op=out):
                _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()
                output_error = self.get_errors_from_gen_backend_stubs(
                    f"backend: PrivateUse1\ncpp_namespace: at::priv1::native\n"
                    f"use_out_as_primary: true\nsupported:\n- {out}"
                )
                self.assertIn("a factory op resolves dtype/layout/device", output_error)

    # Codegen-outcome matrix for out-as-primary across op classes: which registration generates
    # vs raises. Runtime dtype correctness (whether the generated code produces the right dtype) is
    # out of scope here -- it needs the compile+run coverage tracked by the torch_openreg_test_only
    # TODO; this only proves the structural skeleton.
    def test_out_as_primary_codegen_matrix(self) -> None:
        head = "backend: PrivateUse1\ncpp_namespace: at::priv1::native\n"
        oap = "use_out_as_primary: true\n"

        def generates(yaml_str: str) -> None:
            _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()
            self.assert_success_from_gen_backend_stubs(yaml_str)

        def raises(yaml_str: str, needle: str) -> None:
            _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()
            self.assertIn(needle, self.get_errors_from_gen_backend_stubs(yaml_str))

        # (out_name, functional_name, natively_structured)
        cases = [
            ("div.out", "div.Tensor", True),
            ("isin.Tensor_Tensor_out", "isin.Tensor_Tensor", True),
            ("bucketize.Tensor_out", "bucketize.Tensor", False),
            ("sort.values_stable", "sort.stable", True),
            ("_ctc_loss.out", "_ctc_loss", False),
        ]
        for out, func, structured in cases:
            with self.subTest(op=out):
                # Default external (functional primary) and use_out_as_primary + functional
                # registered both always generate.
                generates(f"{head}supported:\n- {func}")
                generates(f"{head}{oap}supported:\n- {func}\n- {out}")
                # use_out_as_primary + structured: true generates iff the op is natively structured.
                structured_yaml = (
                    f"{head}{oap}supported:\n- {out}:\n    structured: true"
                )
                if structured:
                    generates(structured_yaml)
                else:
                    raises(structured_yaml, "is not defined as a structured operator")
                # use_out_as_primary with only the out: a natively-structured op defers its
                # functional to the in-tree composite; a non-structured one is delegated to
                # the kernel with an undefined out. Both generate.
                generates(f"{head}{oap}supported:\n- {out}")

    # structured kernels are out-primary; structured: true without use_out_as_primary would
    # silently emit a plain non-structured out kernel (no meta reuse, no functional), so reject.
    def test_structured_requires_use_out_as_primary(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: false
supported:
- maximum.out:
    structured: true"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Operator 'maximum.out' has 'structured: True' but the backend sets 'use_out_as_primary: False'. Structured kernels are out-primary; set use_out_as_primary: True so the functional/inplace are derived from the out.""",
        )

    # structured: true is only valid on ops that are structured in native_functions.yaml.
    def test_structured_true_on_non_structured_op(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: true
supported:
- abs:
    structured: true"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Operator 'abs' is marked as 'structured: true' in the backend YAML, but it is not defined as a structured operator in native_functions.yaml.""",
        )

    # An unrecognized per-op option key (typo) is treated as a second operator
    # name and rejected, listing the supported option keys.
    def test_per_op_typo_option_key(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: true
supported:
- mul.out:
    structred: true"""  # codespell:ignore structred
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Operator 'mul.out' has unknown option keys ['structred']. Supported option keys: ['device_guard', 'structured'].""",  # codespell:ignore structred
        )

    # The flat dict form (options as siblings of the op name) is a footgun -- one extra space
    # silently turns an option into the op's value -- and is rejected. Options must be nested
    # under the operator name.
    def test_flat_dict_shape_rejected(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
supported:
- mul.out:
  structured: true"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Each 'supported' entry must be a single operator mapping whose value holds the options, but got keys ['mul.out', 'structured']. Indent the options under the operator name.""",
        )


# Golden tests for the generated C++ of the PrivateUse1 out-as-primary /
# structured-kernel feature, mirroring the PR case study (sub.out / mul.out /
# div.out under cpp_namespace at::priv1::native). Following tools/test/test_codegen.py,
# these call the codegen helpers directly and assertExpectedInline the full returned
# string (regenerate with EXPECTTEST_ACCEPT=1) -- no file writing, so no churn from
# the dynamic @generated header.
class TestGenBackendStubsCodegen(expecttest.TestCase):
    def setUp(self) -> None:
        global _GLOBAL_PARSE_NATIVE_YAML_CACHE
        _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()

    _PRIV1_HEADER = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: true
supported:
"""

    # Default functional-primary PrivateUse1 backend (no use_out_as_primary opt-in) -- how an
    # external backend consumes gen_backend_stubs without the structured/out-as-primary opt-in.
    _DEFAULT_PRIV1_HEADER = """\
backend: PrivateUse1
cpp_namespace: priv1::native
supported:
"""

    def _parse(self, supported: str, header: str | None = None):
        # Parse the in-tree native_functions.yaml plus the given PrivateUse1
        # backend stub, and return (groups the backend registers a kernel for,
        # its BackendIndex, the backend class name). This is enough to invoke the
        # codegen helpers directly. The parse cache is cleared each call so the
        # mutated backend_indices does not leak into a later call in the same test.
        global _GLOBAL_PARSE_NATIVE_YAML_CACHE
        _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()
        native = Path(__file__).absolute().parents[2] / "aten/src/ATen/native"
        parsed = parse_native_yaml(
            str(native / "native_functions.yaml"), str(native / "tags.yaml")
        )
        grouped = get_grouped_native_functions(parsed.native_functions)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as fp:
            fp.write((header or self._PRIV1_HEADER) + supported)
            fp.flush()
            backend_yaml = parse_backend_yaml(fp.name, grouped, parsed.backend_indices)
        backend_index = backend_yaml.backend_indices[backend_yaml.backend_key]
        class_name = (
            backend_yaml.class_name or backend_index.native_function_class_name()
        )
        groups = [
            g
            for g in grouped
            if isinstance(g, NativeFunctionsGroup)
            and any(backend_index.has_kernel(f) for f in g.functions())
        ]
        return groups, backend_index, class_name

    def anonymous_definitions(self, supported: str, header: str | None = None) -> str:
        groups, backend_index, class_name = self._parse(supported, header)
        gen = dest.RegisterDispatchKey(
            backend_index,
            Target.ANONYMOUS_DEFINITION,
            SelectiveBuilder.get_nop_selector(),
            rocm=False,
            symint=True,
            class_method_name=class_name,
            skip_dispatcher_op_registration=False,
        )
        code = "\n".join(concatMap(gen, groups))
        # An unused native-AOT consultation slot renders as an indented empty line.
        return "\n".join(line.rstrip() for line in code.split("\n"))

    def native_function_declaration(self, supported: str) -> str:
        groups, backend_index, _ = self._parse(supported)
        return "\n".join(
            concatMap(
                lambda g: dest.compute_native_function_declaration(g, backend_index),
                groups,
            )
        )

    # use_out_as_primary: div.out registered as a plain .out kernel generates an out wrapper that
    # returns the _out call directly (note the at::priv1::native 3-level namespace). Neither the
    # functional nor the inplace is derived: div is natively structured, so both stay with aten's
    # CompositeExplicitAutogradNonFunctional kernels (op.meta() + at::div_outf), which validate
    # and size the output and then redispatch to this backend's div_out -- as on main.
    def test_out_as_primary_wrappers(self) -> None:
        anon = self.anonymous_definitions("- div.out")
        self.assertNotIn("at::_copy_from_and_resize", anon)
        # functional and inplace defer to the composite -- no PrivateUse1 wrapper is emitted
        self.assertNotIn("wrapper_PrivateUse1_Tensor_div(", anon)
        self.assertNotIn("wrapper_PrivateUse1_Tensor_div_(", anon)
        self.assertExpectedInline(
            anon,
            """\
namespace {

at::Tensor & wrapper_PrivateUse1_out_div_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    // No device check


  const OptionalDeviceGuard device_guard(device_of(self));

  return at::priv1::native::PrivateUse1NativeFunctions::div_out(self, other, out);
}

} // anonymous namespace
""",
        )

    # An inplace aten already provides is left to it like the functional (structured div_,
    # composite abs_); one aten has no kernel for (_add_relu_) is derived from the .out and
    # checks devices like any generated wrapper.
    def test_inplace_derived_only_without_aten_kernel(self) -> None:
        anon = self.anonymous_definitions("- div.out\n- abs.out\n- _add_relu.out")
        self.assertNotIn("wrapper_PrivateUse1_Tensor_div_(", anon)
        self.assertNotIn("wrapper_PrivateUse1__abs_(", anon)
        self.assertIn(
            """\
at::Tensor & wrapper_PrivateUse1_Tensor__add_relu_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_PrivateUse1_Tensor__add_relu_", "self");
  c10::impl::check_and_update_common_device(common_device, other, "wrapper_PrivateUse1_Tensor__add_relu_", "other");
  const OptionalDeviceGuard device_guard(device_of(self));
  at::priv1::native::PrivateUse1NativeFunctions::_add_relu_out(self, other, alpha, self);
  return self;
}
""",
            anon,
        )

    # use_out_as_primary only turns off derivation for other external keys, as on main; only
    # PrivateUse1 derives the missing variants from whichever variant it registered.
    def test_use_out_as_primary_derives_nothing_for_other_keys(self) -> None:
        xla = "backend: XLA\ncpp_namespace: torch_xla\nuse_out_as_primary: true\nsupported:\n"
        anon = self.anonymous_definitions("- abs", header=xla)
        self.assertIn("wrapper_XLA__abs(", anon)
        self.assertNotIn("abs_out(", anon)
        self.assertNotIn("abs_(", anon)
        anon = self.anonymous_definitions("- abs")
        self.assertIn("wrapper_PrivateUse1_out_abs_out(", anon)
        self.assertIn("wrapper_PrivateUse1__abs_(", anon)

    # create_out / maybe_create_proxy are only consumed by the structured set_output path, so
    # they are emitted only when a structured op is registered -- not for a default backend,
    # and not for an out-as-primary backend with only plain .out kernels. resize_out and
    # check_inplace are emitted either way.
    def test_default_priv1_omits_structured_set_output_helpers(self) -> None:
        from torchgen.dest.register_dispatch_key import gen_registration_helpers

        _, default_index, _ = self._parse(
            "- mul.Tensor\n- div.out", header=self._DEFAULT_PRIV1_HEADER
        )
        default = "\n".join(gen_registration_helpers(default_index))
        self.assertNotIn("create_out", default)
        self.assertNotIn("maybe_create_proxy", default)
        self.assertIn("resize_out", default)
        self.assertIn("check_inplace", default)

        _, plain_index, _ = self._parse("- div.out")
        plain = "\n".join(gen_registration_helpers(plain_index))
        self.assertNotIn("create_out", plain)
        self.assertNotIn("maybe_create_proxy", plain)

        _, opt_in_index, _ = self._parse("- mul.out:\n    structured: true")
        opt_in = "\n".join(gen_registration_helpers(opt_in_index))
        self.assertIn("create_out", opt_in)
        self.assertIn("maybe_create_proxy", opt_in)

    # structured: true reuses the native meta -- the backend struct inherits the
    # native meta parent and declares only impl().
    def test_structured_native_meta_declaration(self) -> None:
        self.assertExpectedInline(
            self.native_function_declaration("- mul.out:\n    structured: true"),
            """\
struct structured_mul_out : public at::meta::structured_mul_Tensor {
void impl(const at::Tensor & self, const at::Tensor & other, const at::Tensor & out);
};
""",
        )

    # device_guard is active only when the backend-level and per-op guards are
    # both true; a per-op device_guard: false omits the DeviceGuard the op would
    # otherwise inherit from the backend-level device_guard: true.
    def test_per_op_device_guard_override(self) -> None:
        guard = "const OptionalDeviceGuard device_guard(device_of(self));"
        self.assertIn(guard, self.anonymous_definitions("- div.out"))
        # a derived inplace follows the op's device_guard too: no guard and no device check
        derived = self.anonymous_definitions(
            "- _add_relu.out:\n    device_guard: false"
        )
        self.assertNotIn(guard, derived)
        self.assertNotIn("check_and_update_common_device", derived)
        off = self.anonymous_definitions("- div.out:\n    device_guard: false")
        self.assertNotIn(guard, off)
        self.assertExpectedInline(
            off,
            """\
namespace {

at::Tensor & wrapper_PrivateUse1_out_div_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    // No device check


  // DeviceGuard omitted

  return at::priv1::native::PrivateUse1NativeFunctions::div_out(self, other, out);
}

} // anonymous namespace
""",
        )

    # A natively structured op defers its functional to the in-tree composite at any arity:
    # the composite runs op.meta(), which types each output independently (sort -> Long indices).
    def test_multi_output_structured_out_as_primary_defers(self) -> None:
        anon = self.anonymous_definitions("- sort.values_stable")
        self.assertIn("wrapper_PrivateUse1_values_stable_sort_out", anon)
        self.assertNotIn("wrapper_PrivateUse1_stable_sort(", anon)

    # A functional aten already provides -- CompositeExplicitAutograd (abs: at::empty +
    # abs_out) or CompositeImplicitAutograd (adaptive_avg_pool2d, which carries the autograd
    # decomposition) -- is left to it; registering a PrivateUse1 functional would shadow it.
    def test_composite_functional_out_as_primary_defers(self) -> None:
        anon = self.anonymous_definitions("- abs.out\n- adaptive_avg_pool2d.out")
        self.assertIn("wrapper_PrivateUse1_out_abs_out", anon)
        self.assertIn("wrapper_PrivateUse1_out_adaptive_avg_pool2d_out", anon)
        self.assertNotIn("wrapper_PrivateUse1__abs(", anon)
        self.assertNotIn("wrapper_PrivateUse1__adaptive_avg_pool2d(", anon)

    # A non-structured op has no native meta, so its functional is delegated to the kernel:
    # the wrapper passes undefined outs (one per return) for the kernel to allocate and checks
    # that it did, so a kernel that forgets fails here instead of leaking an undefined tensor.
    def test_non_structured_out_as_primary_delegates(self) -> None:
        anon = self.anonymous_definitions("- angle.out\n- native_dropout.out")
        self.assertIn(
            """\
at::Tensor wrapper_PrivateUse1__angle(const at::Tensor & self) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  at::Tensor out;
  at::priv1::native::PrivateUse1NativeFunctions::angle_out(self, out);
  TORCH_CHECK(out.defined(), "angle: out-as-primary kernel must allocate an undefined out");
  return out;
}
""",
            anon,
        )
        self.assertIn(
            """\
::std::tuple<at::Tensor,at::Tensor> wrapper_PrivateUse1__native_dropout(const at::Tensor & input, double p, ::std::optional<bool> train) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, input, "wrapper_PrivateUse1__native_dropout", "input");
  const OptionalDeviceGuard device_guard(device_of(input));
  at::Tensor out0;
  at::Tensor out1;
  at::priv1::native::PrivateUse1NativeFunctions::native_dropout_out(input, p, train, out0, out1);
  TORCH_CHECK(out0.defined(), "native_dropout: out-as-primary kernel must allocate an undefined out");
  TORCH_CHECK(out1.defined(), "native_dropout: out-as-primary kernel must allocate an undefined out");
  return std::make_tuple(out0, out1);
}
""",
            anon,
        )
        # self takes precedence over an earlier positional tensor (where.self: condition, self, other)
        anon = self.anonymous_definitions("- where.self_out")
        self.assertIn(
            "wrapper_PrivateUse1_self_where(const at::Tensor & condition, const at::Tensor & self, "
            "const at::Tensor & other) {\n  // No device check\n"
            "  const OptionalDeviceGuard device_guard(device_of(self));",
            anon,
        )

    # A mutable variant (writes some arguments, returns the rest) is not derivable from the
    # .out; neither codegen pass may emit anything for it, or the registration pass would
    # reference a wrapper the definition pass never defines.
    def test_mutable_variant_not_derived_from_out(self) -> None:
        groups, backend_index, class_name = self._parse("- rrelu_with_noise.out")
        g = next(g for g in groups if g.mutable is not None)
        for target in (Target.ANONYMOUS_DEFINITION, Target.REGISTRATION):
            gen = dest.RegisterDispatchKey(
                backend_index,
                target,
                SelectiveBuilder.get_nop_selector(),
                rocm=False,
                symint=True,
                class_method_name=class_name,
                skip_dispatcher_op_registration=False,
            )
            self.assertIsNone(gen.gen_unstructured(g.mutable, g))

    # TensorList out (runtime-dependent count) can't be pre-allocated to derive the
    # functional, so we emit only the plain out wrapper and leave the functional to the
    # composite (in-tree: split_with_sizes_copy.out works the same way on CUDA).
    def test_out_as_primary_tensorlist_falls_back_to_composite(self) -> None:
        out = self.anonymous_definitions("- unbind_copy.int_out")
        self.assertIn(
            "PrivateUse1NativeFunctions::unbind_copy_out(self, dim, out)", out
        )
        # no derived functional: a Tensor[] output is never allocated as a single empty Tensor
        self.assertNotIn("at::empty", out)

    # Primacy follows the variant the backend registers: under a backend-level
    # use_out_as_primary, an op registered by its functional still derives its out and inplace
    # from that functional (the _copy_from_and_resize path) instead of going unregistered.
    def test_functional_registration_opts_the_op_out_of_out_as_primary(self) -> None:
        out = self.anonymous_definitions("- angle")
        # the functional is the backend kernel, and .out is derived from it
        self.assertIn("PrivateUse1NativeFunctions::angle(self)", out)
        self.assertIn("wrapper_PrivateUse1_out_angle_out", out)
        self.assertIn("at::_copy_from_and_resize", out)

    # The derived wrapper must translate to the .out KERNEL's signature, not the raw dispatcher
    # schema: the schema is always symint-typed, but an .out kernel not declared `symint:` takes
    # IntArrayRef. resize.out carries SymInt[] size, so the derived resize_ has to wrap it in
    # C10_AS_INTARRAYREF_SLOW -- passing the SymIntArrayRef straight through would not compile.
    def test_derived_wrapper_uses_kernel_signature_not_symint_schema(self) -> None:
        out = self.anonymous_definitions("- resize\n- resize.out")
        self.assertIn(
            "PrivateUse1NativeFunctions::resize_out(self, C10_AS_INTARRAYREF_SLOW(size), memory_format, self)",
            out,
        )

    # A structured op that is also a symint op must NOT get a "_symint" suffix on its kernel
    # name: kernel_name becomes the generated struct name (structured_<kernel>), which must
    # match the hand-written TORCH_PRIVATEUSE1_IMPL_FUNC(<op>). SymInt is carried by the
    # meta/impl signature -- in-tree structured structs never carry the suffix.
    def test_structured_symint_kernel_name_has_no_suffix(self) -> None:
        global _GLOBAL_PARSE_NATIVE_YAML_CACHE
        _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()
        native = Path(__file__).absolute().parents[2] / "aten/src/ATen/native"
        parsed = parse_native_yaml(
            str(native / "native_functions.yaml"), str(native / "tags.yaml")
        )
        grouped = get_grouped_native_functions(parsed.native_functions)
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
symint:
- upsample_nearest1d.out
supported:
- upsample_nearest1d.out:
    structured: true"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as fp:
            fp.write(yaml_str)
            fp.flush()
            backend_yaml = parse_backend_yaml(fp.name, grouped, parsed.backend_indices)
        backend_index = backend_yaml.backend_indices[backend_yaml.backend_key]
        metadata = backend_index.index[OperatorName.parse("upsample_nearest1d.out")]
        self.assertTrue(metadata.structured)
        self.assertNotIn("_symint", metadata.kernel)
        self.assertEqual(metadata.kernel, "upsample_nearest1d_out")

    # The <ATen/ops/{op}_meta.h> include is gated on the op's own metadata.structured, not just
    # the aten-native structured-ness. An aten-structured op (div) registered non-structured
    # (out-as-primary only) does not inherit at::meta::structured_div, so its meta header must
    # not be emitted; a structured op (maximum) keeps its include.
    def test_meta_include_gated_on_per_op_structured(self) -> None:
        global _GLOBAL_PARSE_NATIVE_YAML_CACHE
        _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: true
supported:
- maximum.out:
    structured: true
- div.out"""
        with tempfile.TemporaryDirectory() as out_dir:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as fp:
                fp.write(yaml_str)
                fp.flush()
                run(fp.name, out_dir, False)
            header = (Path(out_dir) / "PrivateUse1NativeFunctions.h").read_text()
        self.assertIn("maximum_meta.h", header)
        self.assertNotIn("div_meta.h", header)


if __name__ == "__main__":
    unittest.main()
