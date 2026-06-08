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
from torchgen.gen_backend_stubs import (
    gen_define_meta_registrations,
    parse_backend_yaml,
    run,
)
from torchgen.model import NativeFunctionsGroup, OperatorName
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, Target


# gen_backend_stubs.py is an integration point that is called directly by external backends.
# The tests here are to confirm that badly formed inputs result in reasonable error messages.
class TestGenBackendStubs(expecttest.TestCase):
    def setUp(self) -> None:
        global _GLOBAL_PARSE_NATIVE_YAML_CACHE
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
    # feature) lets an op opt into a structured kernel. The op name and its
    # options (structured/define_meta/device_guard) are siblings, so the
    # options must sit at the same indentation as the "- op:" key.

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

    # define_meta: true (with structured: true) opts into a custom meta.
    def test_valid_per_op_define_meta(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: true
supported:
- sub.out:
    structured: true
    define_meta: true"""
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

    # The full case study from the PR description: structured + custom meta,
    # structured + native meta, and out-as-primary-only with a device_guard override.
    def test_valid_per_op_full_case_study(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: true
supported:
- sub.out:
    structured: true
    define_meta: true
- mul.out:
    structured: true
    define_meta: false
- div.out:
    device_guard: false"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    # define_meta: true requires structured: true.
    def test_define_meta_without_structured(self) -> None:
        yaml_str = """\
backend: PrivateUse1
cpp_namespace: at::priv1::native
use_out_as_primary: true
device_guard: true
supported:
- mul.out:
    define_meta: true"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Operator 'mul.out' has 'define_meta: True' but 'structured: False'. Custom meta functions require a structured kernel.""",
        )

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
            """Operator 'mul.out' has unknown option keys ['structred']. Supported option keys: ['define_meta', 'device_guard', 'structured'].""",  # codespell:ignore structred
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

    def _parse(self, supported: str):
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
            fp.write(self._PRIV1_HEADER + supported)
            fp.flush()
            backend_yaml = parse_backend_yaml(fp.name, grouped, parsed.backend_indices)
        backend_index = backend_yaml.backend_indices[backend_yaml.backend_key]
        class_name = (
            backend_yaml.class_name or backend_index.native_function_class_name()
        )
        groups = [
            g
            for g in grouped
            if isinstance(g, NativeFunctionsGroup) and backend_index.has_kernel(g.out)
        ]
        return groups, backend_index, class_name

    def anonymous_definitions(self, supported: str) -> str:
        groups, backend_index, class_name = self._parse(supported)
        gen = dest.RegisterDispatchKey(
            backend_index,
            Target.ANONYMOUS_DEFINITION,
            SelectiveBuilder.get_nop_selector(),
            rocm=False,
            symint=True,
            class_method_name=class_name,
            skip_dispatcher_op_registration=False,
        )
        return "\n".join(concatMap(gen, groups))

    def native_function_declaration(self, supported: str) -> str:
        groups, backend_index, _ = self._parse(supported)
        return "\n".join(
            concatMap(
                lambda g: dest.compute_native_function_declaration(g, backend_index),
                groups,
            )
        )

    def define_meta_registrations(self, supported: str) -> str:
        groups, backend_index, class_name = self._parse(supported)
        return gen_define_meta_registrations(groups, backend_index, class_name)

    # use_out_as_primary: a non-structured op (div.out) generates an out wrapper
    # that returns the impl _out call directly (note the at::priv1::native 3-level
    # namespace), a functional wrapper that allocates an empty out and reuses it,
    # and an inplace wrapper that feeds self into the out slot -- the old
    # at::_copy_from_and_resize temp-copy is gone.
    def test_out_as_primary_wrappers(self) -> None:
        anon = self.anonymous_definitions("- div.out")
        self.assertNotIn("at::_copy_from_and_resize", anon)
        self.assertExpectedInline(
            anon,
            """\
at::Tensor wrapper_PrivateUse1_Tensor_div(const at::Tensor & self, const at::Tensor & other) {
  const OptionalDeviceGuard device_guard(device_of(self));
  auto out = at::empty({0}, self.options());

  at::priv1::native::PrivateUse1NativeFunctions::div_out(self, other, out);
  return out;
}

namespace {

at::Tensor & wrapper_PrivateUse1_out_div_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    // No device check


  const OptionalDeviceGuard device_guard(device_of(self));
  return at::priv1::native::PrivateUse1NativeFunctions::div_out(self, other, out);
}

} // anonymous namespace

at::Tensor & wrapper_PrivateUse1_Tensor_div_(at::Tensor & self, const at::Tensor & other) {
  const OptionalDeviceGuard device_guard(device_of(self));

  at::priv1::native::PrivateUse1NativeFunctions::div_out(self, other, self);
  return self;
}
""",
        )

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

    # define_meta: true additionally emits `using base` and a custom
    # `void meta(...)` declaration in the struct; without it, neither appears.
    def test_structured_custom_meta_declaration(self) -> None:
        self.assertExpectedInline(
            self.native_function_declaration(
                "- sub.out:\n    structured: true\n    define_meta: true"
            ),
            """\
struct structured_sub_out : public at::meta::structured_sub_Tensor {
// Alias to the base meta class for easy access to native meta logic
using base = at::meta::structured_sub_Tensor;
void meta(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
void impl(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out);
};
""",
        )
        native = self.native_function_declaration("- sub.out:\n    structured: true")
        self.assertNotIn("using base", native)
        self.assertNotIn("void meta(", native)

    # device_guard is active only when the backend-level and per-op guards are
    # both true; a per-op device_guard: false omits the DeviceGuard the op would
    # otherwise inherit from the backend-level device_guard: true.
    def test_per_op_device_guard_override(self) -> None:
        guard = "const OptionalDeviceGuard device_guard(device_of(self));"
        self.assertIn(guard, self.anonymous_definitions("- div.out"))
        off = self.anonymous_definitions("- div.out:\n    device_guard: false")
        self.assertNotIn(guard, off)
        self.assertExpectedInline(
            off,
            """\
at::Tensor wrapper_PrivateUse1_Tensor_div(const at::Tensor & self, const at::Tensor & other) {
  // DeviceGuard omitted
  auto out = at::empty({0}, self.options());

  at::priv1::native::PrivateUse1NativeFunctions::div_out(self, other, out);
  return out;
}

namespace {

at::Tensor & wrapper_PrivateUse1_out_div_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    // No device check


  // DeviceGuard omitted
  return at::priv1::native::PrivateUse1NativeFunctions::div_out(self, other, out);
}

} // anonymous namespace

at::Tensor & wrapper_PrivateUse1_Tensor_div_(at::Tensor & self, const at::Tensor & other) {
  // DeviceGuard omitted

  at::priv1::native::PrivateUse1NativeFunctions::div_out(self, other, self);
  return self;
}
""",
        )

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

    # define_meta: true also registers the backend's meta() under the aten Meta key, so
    # torch.compile / FakeTensor uses the backend's shapes. The generated Meta kernel derives
    # from the backend's structured class (so the backend's overridden meta() runs) and allocates
    # meta tensors in set_output. (Emission-verified; not yet runtime-tested under torch.compile.)
    def test_define_meta_registers_aten_meta_key(self) -> None:
        out = self.define_meta_registrations(
            "- minimum.out:\n    structured: true\n    define_meta: true"
        )
        self.assertExpectedInline(
            out,
            """\
namespace {
struct structured_minimum_out_define_meta final : public at::priv1::native::PrivateUse1NativeFunctions::structured_minimum_out {
    void set_output_raw_strided(
        int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
        at::TensorOptions options, at::DimnameList names
    ) override {
        outputs_[output_idx] = strides.empty()
            ? at::detail::empty_meta(sizes, options.device(at::kMeta))
            : at::detail::empty_strided_meta(sizes, strides, options.device(at::kMeta));
        if (!names.empty()) {
            at::namedinference::propagate_names(outputs_[output_idx], names);
        }
        at::meta::structured_minimum::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    void set_output_strided(
        int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
        at::TensorOptions options, at::DimnameList names
    ) override {
        set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    const at::Tensor & maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<at::Tensor, 1> outputs_;
};
at::Tensor wrapper_Meta_define_minimum(const at::Tensor & self, const at::Tensor & other) {
    structured_minimum_out_define_meta op;
    op.meta(self, other);
    return std::move(op.outputs_[0]);
}
}  // anonymous namespace
TORCH_LIBRARY_IMPL(aten, Meta, m) {
    m.impl("minimum", TORCH_FN(wrapper_Meta_define_minimum));
}
""",
        )

    # define_meta on a non-structured op (no aten structured meta) is not yet supported by the
    # registration generator; it requires emitting a structured-style class with no aten base.
    def test_define_meta_only_structured_for_now(self) -> None:
        # div.out is non-structured here, so no Meta registration is emitted.
        out = self.define_meta_registrations("- div.out")
        self.assertEqual(out, "")

    # A multi-output structured op (sort -> values, indices) must build a tuple in the generated
    # Meta wrapper; returning a single Tensor into the tuple slot would not compile.
    def test_define_meta_multi_output_returns_tuple(self) -> None:
        out = self.define_meta_registrations(
            "- sort.values_stable:\n    structured: true\n    define_meta: true"
        )
        self.assertIn("std::array<at::Tensor, 2> outputs_;", out)
        self.assertIn(
            "return std::make_tuple(std::move(op.outputs_[0]), "
            "std::move(op.outputs_[1]));",
            out,
        )


if __name__ == "__main__":
    unittest.main()
