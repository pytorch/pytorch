from __future__ import annotations

import itertools
import textwrap
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    ConstRefCType,
    CppSignature,
    CppSignatureGroup,
    DispatcherSignature,
    Expr,
    kernel_signature,
    MutRefCType,
    NamedCType,
    NativeSignature,
    tensorT,
)
from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import (
    Argument,
    BackendIndex,
    DeviceCheckType,
    DispatchKey,
    gets_generated_out_inplace_wrapper,
    is_cuda_dispatch_key,
    NativeFunction,
    NativeFunctionsGroup,
    SchemaKind,
    TensorOptionsArguments,
)
from torchgen.utils import assert_never, mapMaybe, Target


if TYPE_CHECKING:
    from torchgen.selective_build.selector import SelectiveBuilder


def gen_registration_headers(
    backend_index: BackendIndex,
    per_operator_headers: bool,
    rocm: bool,
) -> list[str]:
    if per_operator_headers:
        headers = ["#include <ATen/ops/as_strided_native.h>"]
    else:
        headers = ["#include <ATen/NativeFunctions.h>"]

    if backend_index.dispatch_key in (DispatchKey.CPU, DispatchKey.Meta):
        headers.append("#include <ATen/EmptyTensor.h>")
    elif backend_index.dispatch_key == DispatchKey.CUDA:
        if rocm:
            headers.append("#include <ATen/hip/EmptyTensor.h>")
        else:
            headers.append("#include <ATen/cuda/EmptyTensor.h>")
    elif backend_index.dispatch_key == DispatchKey.MPS:
        headers.append("#include <ATen/mps/EmptyTensor.h>")
    elif backend_index.dispatch_key == DispatchKey.XPU:
        # XPU specific, this header resides in third_party/torch-xpu-ops
        headers.append("#include <ATen/xpu/EmptyTensor.h>")
    elif per_operator_headers:
        headers += [
            "#include <ATen/ops/empty.h>",
            "#include <ATen/ops/empty_strided.h>",
            "#include <ATen/ops/_copy_from_and_resize.h>",
            "#include <ATen/ops/_copy_from.h>",
        ]
    else:
        headers.append("#include <ATen/Functions.h>")

    headers.append("#include <c10/macros/Macros.h>")
    return headers


def gen_empty_impl_names(
    backend_index: BackendIndex,
) -> tuple[str | None, str | None]:
    empty_impl = None
    empty_strided_impl = None

    if backend_index.dispatch_key in (
        DispatchKey.Meta,
        DispatchKey.CPU,
        DispatchKey.CUDA,
        DispatchKey.MPS,
        DispatchKey.XPU,
    ):
        dispatch = str(backend_index.dispatch_key).lower()
        empty_impl = f"at::detail::empty_{dispatch}"
        empty_strided_impl = f"at::detail::empty_strided_{dispatch}"
    elif backend_index.dispatch_key in (
        DispatchKey.CompositeExplicitAutogradNonFunctional,
        DispatchKey.QuantizedCPU,
        DispatchKey.QuantizedCUDA,
        DispatchKey.XPU,
    ):
        empty_impl = "at::empty"
        empty_strided_impl = "at::empty_strided"

    return empty_impl, empty_strided_impl


def gen_create_out_helper(backend_index: BackendIndex) -> list[str]:
    if backend_index.dispatch_key == DispatchKey.Meta:
        empty_options = "options.device(at::kMeta)"
    else:
        empty_options = "options"

    empty_impl, empty_strided_impl = gen_empty_impl_names(backend_index)
    if empty_impl is None:
        return []

    return [
        f"""
Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {{
  if (strides.empty()) {{
      return {empty_impl}(sizes, {empty_options});
  }} else {{
      return {empty_strided_impl}(sizes, strides, {empty_options});
  }}
}}
"""
    ]


def gen_maybe_create_proxy_helper(backend_index: BackendIndex) -> list[str]:
    _, empty_strided_impl = gen_empty_impl_names(backend_index)
    return (
        []
        if empty_strided_impl is None
        else [
            f"""
std::optional<Tensor> maybe_create_proxy(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {{
  if (out.strides() != strides) {{
    return {empty_strided_impl}(sizes, strides, options);
  }}
  return std::nullopt;
}}
"""
        ]
    )


def gen_resize_out_helper(backend_index: BackendIndex) -> list[str]:
    if backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
        # The function isn't used by this key (since only functional ops have a kernel for this key),
        # so we need to not include it to avoid a defined-but-not-used error.
        return []
    return [
        """
void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  TORCH_CHECK(options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");
  TORCH_CHECK(options.device() == out.device(),
      "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: avoid the redispatch here
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }
  }
}
"""
    ]


def gen_check_inplace_helper(backend_index: BackendIndex) -> list[str]:
    return [
        """
void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {
  // These checks are needed on those operators that:
  //   1) don't use 'TensorIterator' (e.g. 'addmm' and 'baddbmm')
  //   2) have particular typing rules (e.g. 'cumsum' and 'cumprod')
  // For other operators (e.g. 'add'), 'TensorIterator' already checks
  // these things separately.
  TORCH_CHECK(options.dtype() == self.dtype(),
      "Bad in-place call: ",
      "input tensor dtype ", self.dtype(), " and output tensor dtype ", options.dtype(), " should match");
  TORCH_CHECK(options.device() == self.device(),
      "Bad in-place call: ",
      "input tensor device ", self.device(), " and output tensor device ", options.device(), " should match");
  TORCH_CHECK(sizes == self.sizes(),
      "Bad in-place call: ",
      "input tensor size ", self.sizes(), " and output tensor size ", sizes, " should match");
}
"""
    ]


def gen_registration_helpers(backend_index: BackendIndex) -> list[str]:
    return [
        'C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-function")',
        *gen_create_out_helper(backend_index),
        *gen_resize_out_helper(backend_index),
        *gen_check_inplace_helper(backend_index),
        *gen_maybe_create_proxy_helper(backend_index),
        "C10_DIAGNOSTIC_POP()",
    ]


# Generates Register{dispatch}.cpp (e.g., RegisterCPU.cpp).
#
#   - The primary function of this file is to register all of the
#     implementations for the given dispatch key to the dispatcher,
#     so they are available for use in PyTorch.  If dispatch is
#     None, we generate schema (def) registrations and catchall
#     registrations.
#   - The secondary function of this file is to generate a wrapper
#     around functions.  In CPUType these wrappers do nothing
#     (and should be removed), but in other cases they handle
#     DeviceGuard. A small extra benefit of wrappers is they
#     are not overloaded, so they can be used in the registration
#     API without having to disambiguate which overload you want
#     (as would be the case if you directly registered native::
#     functions).
#   - The tertiary function of this file is to generate *static*
#     cpp API bindings which can be used to bypass dispatcher
#     directly to kernels, but with user-friendly cpp-style API
@dataclass(frozen=True)
class RegisterDispatchKey:
    backend_index: BackendIndex

    target: Literal[
        Target.ANONYMOUS_DEFINITION,
        Target.NAMESPACED_DEFINITION,
        Target.NAMESPACED_DECLARATION,
        Target.REGISTRATION,
    ]

    # Selector object to determine which operators to generate
    # registration code for.
    selector: SelectiveBuilder

    # Whether or not we are actually code-genning for ROCm
    rocm: bool

    # Whether or not to generate symint registrations or not.  External users
    # of codegen who don't care about symints can set this to false to get
    # non-SymInt codegen
    symint: bool

    # The class that all unstructured native functions live under. This is used to improve
    # compiler error messages when a kernel writer adds a native function with the wrong signature.
    # This is only used in unstructured kernels, since structured kernels already live in a class.
    # Finally, this field is currently Optional because it is only used by external backends.
    # It would be nice if we can add the same logic to in-tree kernels too, but that requires updating
    # all of the existing kernel signatures scattered across aten/src/ATen/native.
    class_method_name: str | None

    # Only set to true in lightweight dispatch. If lightweight dispatch is enabled we are registering
    # operators into JIT op registry, thus we need to avoid generating code to register into the dispatcher.
    skip_dispatcher_op_registration: bool

    @staticmethod
    def gen_device_check(
        type: DeviceCheckType, args: list[Argument], method_name: str
    ) -> str:
        if type == DeviceCheckType.NoCheck:
            return "  // No device check\n"

        device_check = "std::optional<Device> common_device = std::nullopt;\n"
        device_check += "(void)common_device; // Suppress unused variable warning\n"
        for arg in args:
            # Only tensor like arguments are eligible
            if arg.type.is_tensor_like():
                device_check += f"""
  c10::impl::check_and_update_common_device(common_device, {arg.name}, "{method_name}", "{arg.name}");"""
        return device_check

    @method_with_native_function
    def __call__(self, f: NativeFunctionsGroup | NativeFunction) -> list[str]:
        if isinstance(f, NativeFunctionsGroup):
            g: NativeFunctionsGroup = f
            # Note: We call gen_structured() if the operator is marked structured, regardless of the backend.
            # gen_structured() has special logic to handle auto-generated kernels.
            if g.structured:
                return self.gen_structured(g)
            else:
                return list(
                    mapMaybe(lambda f: self.gen_unstructured(f, g), g.functions())
                )
        elif isinstance(f, NativeFunction):
            r = self.gen_unstructured(f)
            return [] if r is None else [r]
        else:
            assert_never(f)

    def wrapper_kernel_sig(
        self, f: NativeFunction
    ) -> NativeSignature | DispatcherSignature:
        # The prefix is just to ensure uniqueness. The Dispatcher API doesn't guarantee unique kernel names.
        return DispatcherSignature.from_schema(
            f.func,
            prefix=f"wrapper_{self.backend_index.dispatch_key}_{f.func.name.overload_name}_",
            symint=self.symint,
        )

    def gen_out_inplace_wrapper(
        self, f: NativeFunction, g: NativeFunctionsGroup | None
    ) -> str | None:
        if g is None:
            return None
        k = f.func.kind()
        if k is SchemaKind.inplace:
            copy_op = "at::_copy_from"
        elif k is SchemaKind.out:
            copy_op = "at::_copy_from_and_resize"
        else:
            raise AssertionError("gen_out_inplace_wrapper called on a functional op")

        sig = self.wrapper_kernel_sig(f)
        name = sig.name()

        func_res = f"{name}_tmp"
        return_names = cpp.return_names(f)
        if len(return_names) > 1:
            updates = "\n  ".join(
                f"{copy_op}(std::get<{i}>({func_res}), {ret_name});"
                for i, ret_name in enumerate(return_names)
            )
            returns = f'{sig.returns_type().cpp_type()}({", ".join(return_names)})'
        elif len(return_names) == 1:
            ret_name = return_names[0]
            updates = f"{copy_op}({func_res}, {ret_name});"
            returns = ret_name
        else:
            assert len(f.func.arguments.out) == 1
            returns = ""
            out_arg = f.func.arguments.out[0]
            if out_arg.type.is_list_like():
                updates = f"""\
    for (int64_t i = 0; i < {func_res}.size(); ++i) {{
        {copy_op}({func_res}[i], {out_arg.name}[i]);
    }}"""
            else:
                updates = f"{copy_op}({func_res}, {out_arg.name});"

        functional_sig = self.wrapper_kernel_sig(g.functional)
        wrapper_name = sig.name()

        return f"""\
{sig.defn(name=wrapper_name)} {{
  auto {func_res} = {functional_sig.name()}({", ".join(e.expr for e in translate(sig.arguments(), functional_sig.arguments()))});
  {updates}
  return {returns};
}}
"""

    def gen_structured(self, g: NativeFunctionsGroup) -> list[str]:
        metadata = self.backend_index.get_kernel(g)
        if self.backend_index.dispatch_key == DispatchKey.Meta:
            assert not self.backend_index.has_kernel(g.out), (
                "Do not explicitly specify Meta dispatch key on structured "
                "functions, they will be automatically generated for you"
            )
        elif (
            self.backend_index.dispatch_key
            == DispatchKey.CompositeExplicitAutogradNonFunctional
        ):
            assert not self.backend_index.has_kernel(g.out), (
                "Do not explicitly specify CompositeExplicitAutograd dispatch key on structured "
                "functions, they will be automatically generated for you"
            )
        elif metadata is None or not metadata.structured:
            return list(mapMaybe(lambda f: self.gen_unstructured(f, g), g.functions()))
        structured_gen = StructuredRegisterDispatchKey(
            self.backend_index,
            self.target,
            self.selector,
            self.rocm,
            self.symint,
            self.class_method_name,
            self.skip_dispatcher_op_registration,
            g,
        )
        return list(mapMaybe(structured_gen.gen_one, g.functions()))

    def gen_unstructured(
        self, f: NativeFunction, g: NativeFunctionsGroup | None = None
    ) -> str | None:
        with native_function_manager(f):
            inplace_meta = False
            gets_out_inplace_wrapper = False
            if not self.backend_index.has_kernel(f):
                if (
                    self.backend_index.dispatch_key == DispatchKey.Meta
                    and f.func.kind() is SchemaKind.inplace
                    and
                    # Defer to composites for meta implementation
                    not f.has_composite_kernel
                    and
                    # Inplace list operations are not supported
                    len(f.func.returns) == 1
                ):
                    inplace_meta = True
                elif (
                    not self.backend_index.use_out_as_primary
                    and g is not None
                    and gets_generated_out_inplace_wrapper(f, g, self.backend_index)
                ):
                    # We want to generate inplace/out wrappers, that don't have a kernel for the backend.
                    gets_out_inplace_wrapper = True
                else:
                    return None
            if f.manual_kernel_registration:
                return None

            if (
                self.target is Target.REGISTRATION
                and not self.selector.is_native_function_selected(f)
            ):
                return None

            sig = self.wrapper_kernel_sig(f)

            name = sig.name()
            returns_type = sig.returns_type().cpp_type()
            args = sig.arguments()
            args_str = ", ".join(a.defn() for a in args)

            # See Note [Direct dispatch bindings]
            cpp_sig_group = CppSignatureGroup.from_native_function(
                f, method=False, fallback_binding=False
            )

            # TODO: dedupe this with the structured codegen
            if self.target is Target.NAMESPACED_DECLARATION:
                result = ""
                for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                    result += f"TORCH_API {cpp_sig.decl()};\n"
                return result
            elif self.target is Target.NAMESPACED_DEFINITION:

                def generate_defn(cpp_sig: CppSignature) -> str:
                    return f"""
{cpp_sig.defn()} {{
return {sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
"""

                result = ""
                for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                    result += generate_defn(cpp_sig)
                return result

            elif self.target is Target.ANONYMOUS_DEFINITION:
                # short circuit for inplace_meta
                if inplace_meta:
                    assert f.func.arguments.self_arg is not None
                    self_arg_name = f.func.arguments.self_arg.argument.name
                    # TODO: handle in place on tensor list
                    return f"""
{returns_type} {name}({args_str}) {{
  TORCH_CHECK_NOT_IMPLEMENTED({self_arg_name}.is_meta(),
    "Cannot inplace into non-meta tensor with meta tensor argument");
  return {self_arg_name};
}}
"""

                # short circuit for generated inplace/out wrappers
                if gets_out_inplace_wrapper:
                    return self.gen_out_inplace_wrapper(f, g)

                metadata = self.backend_index.get_kernel(f)
                if metadata is None:
                    return None
                if self.class_method_name is None:
                    impl_name = f"{metadata.cpp_namespace}::{metadata.kernel}"
                else:
                    impl_name = f"{metadata.cpp_namespace}::{self.class_method_name}::{metadata.kernel}"

                kernel_sig = kernel_signature(f, self.backend_index)

                args_exprs_str = ", ".join(
                    e.expr
                    for e in translate(
                        sig.arguments(), kernel_sig.arguments(), method=False
                    )
                )

                device_check = "  // No device check\n"
                # Backends that require device guards presumably also require device checks.
                if self.backend_index.device_guard:
                    device_check_args = itertools.chain(
                        f.func.arguments.out, f.func.arguments.flat_positional
                    )
                    device_check = RegisterDispatchKey.gen_device_check(
                        f.device_check, list(device_check_args), name
                    )

                device_guard = "// DeviceGuard omitted"  # default
                if f.device_guard and self.backend_index.device_guard:
                    has_tensor_options = any(
                        isinstance(a, TensorOptionsArguments)
                        for a in f.func.arguments.non_out
                    )
                    if has_tensor_options:
                        # kernel is creating a tensor
                        device_guard = """
  const DeviceGuard device_guard(device_or_default(device));"""

                        # CUDA requires special handling
                        if is_cuda_dispatch_key(self.backend_index.dispatch_key):
                            device_guard = f"globalContext().lazyInitDevice(c10::DeviceType::CUDA);\n{device_guard}"
                    else:
                        # kernel is operating on existing tensors

                        # There is precedence for which argument we use to do
                        # device guard.  This describes the precedence order.
                        self_arg = (
                            [f.func.arguments.self_arg.argument]
                            if f.func.arguments.self_arg is not None
                            else []
                        )
                        candidate_args = itertools.chain(
                            self_arg,
                            f.func.arguments.out,
                            f.func.arguments.flat_positional,
                        )

                        # Only tensor like arguments are eligible
                        device_of = next(
                            (
                                f"{a.name}"
                                for a in candidate_args
                                if a.type.is_tensor_like()
                            ),
                            None,
                        )
                        if device_of is not None:
                            device_guard = f"const OptionalDeviceGuard device_guard(device_of({device_of}));"

                return f"""\
namespace {{

{returns_type} {name}({args_str}) {{
  {device_check}

  {device_guard}
  return {impl_name}({args_exprs_str});
}}

}} // anonymous namespace
"""

            elif self.target is Target.REGISTRATION:
                if f.manual_kernel_registration or self.skip_dispatcher_op_registration:
                    return None
                else:
                    payload = f"TORCH_FN({name})"
                    return f'm.impl("{f.func.name}",\n{payload});\n'
            else:
                assert_never(self.target)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           STRUCTURED
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@dataclass(frozen=True)
class StructuredRegisterDispatchKey(RegisterDispatchKey):
    g: NativeFunctionsGroup

    def gen_class_set_output_functions(
        self, k: SchemaKind, parent_class: str, generate_super: bool
    ) -> str:
        if generate_super:
            set_output_super = f"{parent_class}::set_output_raw_strided(output_idx, sizes, strides, options, names);"
        else:
            set_output_super = ""

        def gen_set_output_function(name: str, maybe_create_proxy: bool) -> str:
            return f"""
void set_output_{name}(
    int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
    TensorOptions options, DimnameList names
) override {{
{textwrap.indent(self.gen_class_set_output_body(k, maybe_create_proxy), "    ")}
    if (!names.empty()) {{
      namedinference::propagate_names(outputs_[output_idx], names);
    }}
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
{textwrap.indent(set_output_super, "    ")}
}}
"""

        return f"""
{gen_set_output_function("strided", maybe_create_proxy=True)}
{gen_set_output_function("raw_strided", maybe_create_proxy=False)}
"""

    def gen_class_set_output_body(self, k: SchemaKind, maybe_create_proxy: bool) -> str:
        if self.backend_index.dispatch_key in [
            DispatchKey.CUDA,
            DispatchKey.MPS,
            DispatchKey.XPU,
            DispatchKey.CompositeExplicitAutogradNonFunctional,
        ]:
            maybe_set_guard = """
auto current_device = guard_.current_device();
if (C10_UNLIKELY(current_device.has_value())) {
  TORCH_INTERNAL_ASSERT(*current_device == options.device(),
    "structured kernels don't support multi-device outputs");
} else {
  guard_.reset_device(options.device());
}
"""
            maybe_set_guard_line = maybe_set_guard + "\n"
        else:
            maybe_set_guard_line = maybe_set_guard = ""

        if maybe_create_proxy:
            create_proxy = """
auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
if (C10_UNLIKELY(maybe_proxy.has_value())) {
    proxy_outputs_[output_idx] = std::move(maybe_proxy).value();
}
"""
        else:
            create_proxy = ""

        if k is SchemaKind.functional:
            assert self.backend_index.dispatch_key in (
                DispatchKey.Meta,
                DispatchKey.CPU,
                DispatchKey.CUDA,
                DispatchKey.MPS,
                DispatchKey.XPU,
                DispatchKey.CompositeExplicitAutogradNonFunctional,
            )
            return f"""{maybe_set_guard_line}
outputs_[output_idx] = create_out(sizes, strides, options);"""
        elif k is SchemaKind.inplace:
            return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
check_inplace(out, sizes, options);
{create_proxy}"""
        elif k is SchemaKind.out:
            return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
resize_out(out, sizes, strides, options);
{create_proxy}"""
        elif k is SchemaKind.mutable or k is SchemaKind.scratch:
            raise AssertionError(
                f"{k} structured operators are currently not supported"
            )
        else:
            assert_never(k)

    # returns the definition of a ctor, as well as how to construct
    # this class to a variable named op
    def gen_class_ctor(self, k: SchemaKind, class_name: str, returns: int) -> str:
        if k is SchemaKind.functional:
            return ""
        elif k is SchemaKind.inplace:
            # TODO: Make sure out argument is guaranteed to be self
            return f"{class_name}(Tensor& self) : outputs_{{std::ref(self)}} {{}}"
        elif k is SchemaKind.out:
            out_args = ", ".join(f"Tensor& out{i}" for i in range(returns))
            out_refs = ", ".join(f"std::ref(out{i})" for i in range(returns))
            return f"{class_name}({out_args}) : outputs_{{ {out_refs} }} {{}}"
        elif k is SchemaKind.mutable or k is SchemaKind.scratch:
            raise AssertionError(
                f"{k} structured operators are currently not supported"
            )
        else:
            assert_never(k)

    def gen_class(
        self,
        f: NativeFunction,
        k: SchemaKind,
        *,
        class_name: str,
        parent_class: str,
        generate_super: bool,
    ) -> str:
        if k is SchemaKind.functional:
            output_type = "Tensor"
            output_value = "outputs_[output_idx]"
            proxy_field = ""
        elif k is SchemaKind.inplace:
            output_type = "std::reference_wrapper<Tensor>"
            output_value = "proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()"
            proxy_field = f"std::array<::std::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;"
        elif k is SchemaKind.out:
            output_type = "std::reference_wrapper<Tensor>"
            output_value = "proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()"
            proxy_field = f"std::array<::std::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;"
        else:
            raise RuntimeError(f"Unsupported SchemaKind {k}")

        if self.backend_index.dispatch_key == DispatchKey.CUDA:
            if self.rocm:
                guard_field = "c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard_;"
            else:
                guard_field = "c10::cuda::OptionalCUDAGuard guard_;"
        elif (
            self.backend_index.dispatch_key
            == DispatchKey.CompositeExplicitAutogradNonFunctional
        ):
            guard_field = "c10::OptionalDeviceGuard guard_;"
        elif self.backend_index.dispatch_key == DispatchKey.MPS:
            # TODO: Move to OptionalMPSGuard.
            guard_field = "c10::OptionalDeviceGuard guard_;"
        elif self.backend_index.dispatch_key == DispatchKey.XPU:
            guard_field = "c10::OptionalDeviceGuard guard_;"
        else:
            guard_field = ""

        indent = " " * 4
        class_ctor_str = self.gen_class_ctor(k, class_name, len(f.func.returns))
        lines = (
            f"struct {class_name} final : public {parent_class} {{",
            f"{textwrap.indent(class_ctor_str, indent)}",
            f"{textwrap.indent(self.gen_class_set_output_functions(k, parent_class, generate_super), indent)}",
            "    const Tensor& maybe_get_output(int64_t output_idx) override {",
            f"      return {output_value};\n",  # type: ignore[possibly-undefined]  # TODO: audit
            "    }",
            # type: ignore[possibly-undefined]  # TODO: audit
            f"    std::array<{output_type}, {len(f.func.returns)}> outputs_;",
            f"{textwrap.indent(proxy_field, indent)}",  # type: ignore[possibly-undefined]  # TODO: audit
            f"{textwrap.indent(guard_field, indent)}",
            "};",
        )
        return "\n".join(line for line in lines if line)

    @method_with_native_function
    def gen_one(self, f: NativeFunction) -> str | None:
        assert not f.manual_kernel_registration

        if (
            self.target is Target.REGISTRATION
            and not self.selector.is_native_function_selected(f)
        ):
            return None

        # TODO: Now, there is something interesting going on here.  In the code below,
        # we generate CompositeExplicitAutogradNonFunctional implementations of functional and inplace
        # based on the out implementation.  But in fact, out is definable by
        # functional too (just not very efficiently), and this is honestly the
        # MORE likely situation for a backend implementor.  How do we pick?
        # Well, taking a page from Haskell type classes and default methods,
        # we could conceivably register a circular definition (out in terms
        # of functional, and functional in terms of out) and just require
        # someone to implement one or the other.  We'd have to do a little bit
        # of work to not register one of these "weak" definitions unless there
        # is a strong definition somewhere in the DAG!  So it's not implemented yet.
        if (
            self.backend_index.dispatch_key
            == DispatchKey.CompositeExplicitAutogradNonFunctional
            and f.func.kind() is SchemaKind.out
        ):
            # Never generate a default implementation for out, that's what you
            # have to define as a backend implementor
            return None

        # Note [Direct dispatch bindings]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Signature of the non-dispatched function we'll expose in a header
        # (e.g., at::cpu::add).  We don't generate methods (TODO: do this
        # when CPUTensor class is a thing); nor do we generate fallback
        # bindings for manual_cpp_binding functions.
        cpp_sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=False
        )

        # Signature of the wrapper function we'll register to the dispatcher
        kern = self.backend_index.get_kernel(f)
        sig = NativeSignature(
            f.func,
            prefix=f"wrapper_{self.backend_index.dispatch_key}_",
            symint=kern is not None and kern.supports_symint(),
        )

        if self.target is Target.NAMESPACED_DECLARATION:
            result = ""
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += f"TORCH_API {cpp_sig.decl()};\n"
            return result

        elif self.target is Target.NAMESPACED_DEFINITION:

            def generate_defn(cpp_sig: CppSignature) -> str:
                return f"""
{cpp_sig.defn()} {{
return {sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
"""

            result = ""
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += generate_defn(cpp_sig)
            return result

        elif self.target is Target.ANONYMOUS_DEFINITION:
            k = f.func.kind()

            # Construct the body of the wrapper function with signature sig
            sig_body = []
            # We'll use context to keep track of any variables we've brought
            # into scope while generating code
            context: list[Binding | Expr] = list(sig.arguments())

            # Initialize the class corresponding to this structured
            # operator; feeding it the output argument(s) if it is known
            if self.backend_index.dispatch_key is DispatchKey.Meta:
                class_name = f"structured_{meta.name(self.g)}_meta_{k.name}"
                parent_class = f"at::meta::structured_{meta.name(self.g)}"
            elif (
                self.backend_index.dispatch_key
                is DispatchKey.CompositeExplicitAutogradNonFunctional
            ):
                # TODO: dedup this branch
                class_name = f"structured_{meta.name(self.g)}_default_backend_{k.name}"
                parent_class = f"at::meta::structured_{meta.name(self.g)}"
            else:
                metadata = self.backend_index.get_kernel(self.g)
                assert metadata is not None
                class_name = f"structured_{metadata.kernel}_{k.name}"
                parent_class = f"{metadata.cpp_namespace}::structured_{metadata.kernel}"

            if self.backend_index.device_guard:
                device_check_args = itertools.chain(
                    f.func.arguments.out, f.func.arguments.flat_positional
                )
                sig_body.append(
                    RegisterDispatchKey.gen_device_check(
                        f.device_check, list(device_check_args), sig.name()
                    )
                )

            if k is SchemaKind.functional:
                sig_body.append(f"{class_name} op;")
            elif k is SchemaKind.inplace:
                sig_body.append(f"{class_name} op(self);")
            elif k is SchemaKind.out:
                out_args_str = ", ".join(a.name for a in f.func.arguments.out)
                sig_body.append(f"{class_name} op({out_args_str});")

            # Translate the input native arguments into structured
            # arguments for the meta call
            meta_exprs = ", ".join(
                e.expr
                for e in translate(
                    context, structured.meta_arguments(self.g), method=False
                )
            )

            if self.g.out.precomputed:
                # If this function group has precomputed elements, the meta function
                # returns a struct containing them which must be saved so that it
                # can be unpacked when generating code to call the impl.
                sig_body.append(f"auto precompute = op.meta({meta_exprs});")

                # Put all of the contents of the precompute struct into the context
                # so that translate will be able to return the correct args for the
                # call to the impl.
                precomputed_values = [
                    *self.g.out.precomputed.replace.values(),
                    self.g.out.precomputed.add,
                ]
                for precomputed_elems in precomputed_values:
                    context.extend(
                        Expr(
                            expr=f"precompute.{arg.name}",
                            type=structured.argument_type(arg, binds=arg.name),
                        )
                        for arg in precomputed_elems
                    )

                # Add a use of the precompute struct so FB internal compilers don't
                # complain that there is an unused variable.
                sig_body.append("(void)precompute;")
            else:
                sig_body.append(f"op.meta({meta_exprs});")

            # After running meta, op.outputs_ is guaranteed to be valid;
            # add it to the context
            out_args = structured.out_arguments(self.g)
            for i, out_arg in enumerate(out_args):
                assert ConstRefCType(BaseCType(tensorT)) == out_arg.nctype.type

                if k is SchemaKind.out:
                    expr = f"op.maybe_get_output({i})"
                else:
                    expr = f"op.outputs_[{i}]"

                context.append(
                    Expr(
                        expr=expr,
                        # TODO: Stop hardcoding that the output type is a Tensor.  Note
                        # that for the codegen here this is fine because outputs_ is
                        # hardcoded to be tensor already
                        type=NamedCType(
                            out_arg.nctype.name, MutRefCType(BaseCType(tensorT))
                        ),
                    )
                )

            # With the expanded context, do the impl call (if not a meta
            # function)
            if (
                self.backend_index.dispatch_key
                == DispatchKey.CompositeExplicitAutogradNonFunctional
            ):
                # TODO: https://github.com/pytorch/pytorch/issues/53023
                out_sig_group = CppSignatureGroup.from_native_function(
                    self.g.out, method=False, fallback_binding=f.manual_cpp_binding
                )
                out_sig = out_sig_group.most_faithful_signature()
                api_name = out_sig.name()
                out_exprs = ", ".join(
                    e.expr
                    for e in translate(context, out_sig.arguments(), method=False)
                )
                # TODO: I think this means structured won't work with method
                # only functions (but maybe you're saved by faithful? iunno.)
                # NB: Originally I wrote this as an at::redispatch call, but
                # I got in trouble because that meant I needed a DispatchKeySet
                # in the wrapper function, which meant I needed a DispatchKeySet
                # in the DispatchKeyFunctions declarations, but the defined API
                # there does NOT permit a dispatch key set.  I think you can
                # probably unwind this by calling some function to do the TLS
                # fetch and get the DispatchKeySet when you don't have it, but
                # I didn't do it for this version
                sig_body.append(f"at::{api_name}({out_exprs});")
            elif self.backend_index.dispatch_key != DispatchKey.Meta:
                impl_exprs = ", ".join(
                    e.expr
                    for e in translate(
                        context, structured.impl_arguments(self.g), method=False
                    )
                )
                sig_body.append(f"op.impl({impl_exprs});")

            # Go over each output, and check if there is a proxy created for it.
            # If so, copy it over to the original output.
            if k is SchemaKind.out or k is SchemaKind.inplace:
                for i in range(len(f.func.returns)):
                    sig_body.append(
                        f"if (op.proxy_outputs_[{i}].has_value()) op.outputs_[{i}].get().copy_(*op.proxy_outputs_[{i}]);"
                    )

            # Destructively return the final tensors
            # TODO: Do this in translate instead
            if k is SchemaKind.functional:
                if len(f.func.returns) == 1:
                    ret_expr = "std::move(op.outputs_[0])"  # small optimization
                else:
                    moved = ", ".join(
                        f"std::move(op.outputs_[{i}])"
                        for i in range(len(f.func.returns))
                    )
                    ret_expr = f"std::make_tuple({moved})"
            elif k is SchemaKind.inplace:
                ret_expr = "self"
            elif k is SchemaKind.out:
                if len(f.func.returns) == 1:
                    ret_expr = f.func.arguments.out[0].name
                else:
                    refs = ", ".join(a.name for a in f.func.arguments.out)
                    ret_expr = f"std::forward_as_tuple({refs})"
            sig_body.append(f"return {ret_expr};")  # type: ignore[possibly-undefined]  # TODO: audit

            sig_body_str = "\n".join(sig_body)

            # For an overview of what this template code looks like, see
            # https://github.com/pytorch/rfcs/pull/9
            return f"""\
{self.gen_class(
f, k,
class_name=class_name,
parent_class=parent_class,
generate_super=self.g.out.structured_inherits is not None
)}

{sig.defn()} {{
{sig_body_str}
}}
"""

        elif self.target is Target.REGISTRATION:
            return f'm.impl("{f.func.name}", TORCH_FN({sig.name()}));'
        else:
            assert_never(self.target)
            # Silence mypy's "Missing return statement" error
            return None
