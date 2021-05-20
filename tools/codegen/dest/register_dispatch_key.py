from typing import List, Optional, Union
import itertools
from typing_extensions import Literal
from dataclasses import dataclass
import textwrap

from tools.codegen.context import method_with_native_function
from tools.codegen.utils import Target, mapMaybe
from tools.codegen.model import (DispatchKey, NativeFunction,
                                 NativeFunctionsGroup, SchemaKind,
                                 TensorOptionsArguments,
                                 DeviceCheckType, Argument,
                                 assert_never,
                                 is_cuda_dispatch_key,
                                 is_structured_dispatch_key)
from tools.codegen.api.types import (BaseCType, Binding, ConstRefCType,
                                     CppSignature, CppSignatureGroup,
                                     DispatcherSignature, Expr, MutRefCType,
                                     NativeSignature, tensorT, NamedCType)
import tools.codegen.api.meta as meta
import tools.codegen.api.structured as structured
from tools.codegen.api.translate import translate
from tools.codegen.selective_build.selector import SelectiveBuilder

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
    dispatch_key: DispatchKey

    target: Union[
        Literal[Target.ANONYMOUS_DEFINITION],
        Literal[Target.NAMESPACED_DEFINITION],
        Literal[Target.NAMESPACED_DECLARATION],
        Literal[Target.REGISTRATION]
    ]

    # Selector object to determine which operators to generate
    # registration code for.
    selector: SelectiveBuilder

    # Whether or not we are actually code-genning for ROCm
    rocm: bool

    @staticmethod
    def gen_device_check(type: DeviceCheckType, args: List[Argument], method_name: str) -> str:
        if type == DeviceCheckType.NoCheck:
            return '  // No device check\n'

        device_check = 'c10::optional<Device> common_device = nullopt;'
        for arg in args:
            # Only tensor like arguments are eligible
            if arg.type.is_tensor_like():
                device_check += f"""
  c10::impl::check_and_update_common_device(common_device, {arg.name}, "{method_name}", "{arg.name}");"""
        return device_check

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        if isinstance(f, NativeFunctionsGroup):
            if f.structured:
                return self.gen_structured(f)
            else:
                return list(mapMaybe(self.gen_unstructured, f.functions()))
        elif isinstance(f, NativeFunction):
            r = self.gen_unstructured(f)
            return [] if r is None else [r]
        else:
            assert_never(f)

    def gen_structured(self, g: NativeFunctionsGroup) -> List[str]:
        if self.dispatch_key == DispatchKey.Meta:
            assert self.dispatch_key not in g.out.dispatch, \
                "Do not explicitly specify Meta dispatch key on structured " \
                "functions, they will be automatically generated for you"
        elif self.dispatch_key == DispatchKey.CompositeExplicitAutograd:
            assert self.dispatch_key not in g.out.dispatch, \
                "Do not explicitly specify CompositeExplicitAutograd dispatch key on structured " \
                "functions, they will be automatically generated for you"
        elif not is_structured_dispatch_key(self.dispatch_key):
            return list(mapMaybe(self.gen_unstructured, g.functions()))
        elif self.dispatch_key not in g.out.dispatch:
            return []

        structured_gen = StructuredRegisterDispatchKey(
            self.dispatch_key,
            self.target,
            self.selector,
            self.rocm,
            g
        )
        return list(mapMaybe(structured_gen.gen_one, g.functions()))

    @method_with_native_function
    def gen_unstructured(self, f: NativeFunction) -> Optional[str]:
        inplace_meta = False
        if self.dispatch_key not in f.dispatch:
            if (self.dispatch_key == DispatchKey.Meta and
                    f.func.kind() is SchemaKind.inplace and
                    # Defer to composites for meta implementation
                    DispatchKey.CompositeImplicitAutograd not in f.dispatch and
                    DispatchKey.CompositeExplicitAutograd not in f.dispatch and
                    # Inplace list operations are not supported
                    len(f.func.returns) == 1):
                inplace_meta = True
            else:
                return None
        if f.manual_kernel_registration:
            return None

        if self.target is Target.REGISTRATION and not self.selector.is_native_function_selected(f):
            return None

        sig = NativeSignature(f.func, prefix='wrapper_')

        name = sig.name()
        returns_type = sig.returns_type().cpp_type()
        args = sig.arguments()
        args_str = ', '.join(a.defn() for a in args)

        # See Note [Direct dispatch bindings]
        cpp_sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)

        if self.target is Target.NAMESPACED_DECLARATION:
            result = f"TORCH_API {cpp_sig_group.signature.decl()};\n"
            if cpp_sig_group.faithful_signature is not None:
                result += f"TORCH_API {cpp_sig_group.faithful_signature.decl()};\n"
            return result
        elif self.target is Target.NAMESPACED_DEFINITION:
            def generate_defn(cpp_sig: CppSignature) -> str:
                return f"""
{cpp_sig.defn()} {{
return {sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
"""
            result = generate_defn(cpp_sig_group.signature)
            if cpp_sig_group.faithful_signature is not None:
                result += generate_defn(cpp_sig_group.faithful_signature)
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

            impl_name = f"at::native::{f.dispatch[self.dispatch_key]}"

            args_exprs_str = ', '.join(a.name for a in args)

            device_check = '  // No device check\n'
            if is_cuda_dispatch_key(self.dispatch_key):
                device_check_args = itertools.chain(
                    f.func.arguments.out,
                    f.func.arguments.flat_positional
                )
                device_check = RegisterDispatchKey.gen_device_check(f.device_check, list(device_check_args), name)

            device_guard = "// DeviceGuard omitted"  # default
            if f.device_guard and is_cuda_dispatch_key(self.dispatch_key):
                has_tensor_options = any(isinstance(a.argument, TensorOptionsArguments) for a in args)
                if has_tensor_options:
                    # kernel is creating a tensor
                    device_guard = """globalContext().lazyInitCUDA();
  const DeviceGuard device_guard(device_or_default(device));"""
                else:
                    # kernel is operating on existing tensors

                    # There is precedence for which argument we use to do
                    # device guard.  This describes the precedence order.
                    self_arg = [f.func.arguments.self_arg.argument] if f.func.arguments.self_arg is not None else []
                    candidate_args = itertools.chain(
                        self_arg,
                        f.func.arguments.out,
                        f.func.arguments.flat_positional
                    )

                    # Only tensor like arguments are eligible
                    device_of = next((f'{a.name}' for a in candidate_args if a.type.is_tensor_like()), None)
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
            if f.manual_kernel_registration:
                return None
            else:
                dispatcher_sig = DispatcherSignature.from_schema(f.func)
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

    def gen_class_set_output(self, k: SchemaKind, parent_class: str, generate_super: bool) -> str:
        if generate_super:
            set_output_super = f"{parent_class}::set_output(output_idx, sizes, strides, options, names);"
        else:
            set_output_super = ""
        return f"""
void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                TensorOptions options, DimnameList names) override {{
{textwrap.indent(self.gen_class_set_output_body(k), "    ")}
    if (!names.empty()) {{
      namedinference::propagate_names(outputs_[output_idx], names);
    }}
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
{textwrap.indent(set_output_super, "    ")}
}}
"""

    def gen_class_set_output_body(self, k: SchemaKind) -> str:
        if self.dispatch_key in [DispatchKey.CUDA, DispatchKey.CompositeExplicitAutograd]:
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
            maybe_set_guard_line = maybe_set_guard = ''

        if k is SchemaKind.functional:
            if self.dispatch_key == DispatchKey.Meta:
                # TODO: dedupe this with below
                return """
if (strides.empty()) {
    outputs_[output_idx] = at::empty(sizes, options.device(at::kMeta));
} else {
    outputs_[output_idx] = at::empty_strided(sizes, strides, options.device(at::kMeta));
}
"""
            else:
                expanded_topts = "optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), " \
                    "options.device_opt(), options.pinned_memory_opt()"
                if self.dispatch_key == DispatchKey.CPU:
                    empty_impl = "at::native::empty_cpu"
                    empty_strided_impl = "at::native::empty_strided_cpu"
                elif self.dispatch_key == DispatchKey.CUDA:
                    empty_impl = "at::native::empty_cuda"
                    empty_strided_impl = "at::native::empty_strided_cuda"
                elif self.dispatch_key == DispatchKey.CompositeExplicitAutograd:
                    empty_impl = "at::empty"
                    empty_strided_impl = "at::empty_strided"
                else:
                    raise AssertionError("unsupported dispatch key")
                return f"""{maybe_set_guard_line}
if (strides.empty()) {{
    outputs_[output_idx] = {empty_impl}(sizes, {expanded_topts}, options.memory_format_opt());
}} else {{
    // TODO: assert options.memory_format_opt() is nullopt (debug only?)
    outputs_[output_idx] = {empty_strided_impl}(sizes, strides, {expanded_topts});
}}
"""
        elif k is SchemaKind.inplace:
            return maybe_set_guard
        elif k is SchemaKind.out:
            return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
TORCH_CHECK(options.dtype() == out.dtype(),
    "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");
TORCH_CHECK(options.device() == out.device(),
    "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");
bool resized = at::native::resize_output(outputs_[output_idx], sizes);
// Only restride if a resize occurred; otherwise we ignore the (advisory)
// strides from the meta function and directly use the output tensor's
// preexisting strides
if (resized) {{
    if (!strides.empty()) {{
        TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
        at::native::as_strided_(outputs_[output_idx], sizes, strides);
    }} else if (options.memory_format_opt().has_value()) {{
        outputs_[output_idx].get().unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }}
}}
"""
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
            out_args = ', '.join(f"Tensor& out{i}" for i in range(returns))
            out_refs = ', '.join(f"std::ref(out{i})" for i in range(returns))
            return f"{class_name}({out_args}) : outputs_{{ {out_refs} }} {{}}"
        else:
            assert_never(k)

    def gen_class(
        self, f: NativeFunction, k: SchemaKind, *, class_name: str, parent_class: str, generate_super: bool
    ) -> str:
        if k is SchemaKind.functional:
            output_type = "Tensor"
        elif k is SchemaKind.inplace:
            output_type = "std::reference_wrapper<Tensor>"
        elif k is SchemaKind.out:
            output_type = "std::reference_wrapper<Tensor>"

        if self.dispatch_key == DispatchKey.CUDA:
            if self.rocm:
                guard_field = 'c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard_;'
            else:
                guard_field = 'c10::cuda::OptionalCUDAGuard guard_;'
        elif self.dispatch_key == DispatchKey.CompositeExplicitAutograd:
            guard_field = 'c10::OptionalDeviceGuard guard_;'
        else:
            guard_field = ''

        indent = " " * 4
        class_ctor_str = self.gen_class_ctor(k, class_name, len(f.func.returns))
        lines = (
            f"struct {class_name} final : public {parent_class} {{",
            f"{textwrap.indent(class_ctor_str, indent)}",
            f"{textwrap.indent(self.gen_class_set_output(k, parent_class, generate_super), indent)}",
            "    const Tensor& maybe_get_output(int64_t output_idx) override {",
            "        return outputs_[output_idx];",
            "    }",
            f"    std::array<{output_type}, {len(f.func.returns)}> outputs_;",
            f"{textwrap.indent(guard_field, indent)}",
            "};"
        )
        return '\n'.join(line for line in lines if line)

    @method_with_native_function
    def gen_one(self, f: NativeFunction) -> Optional[str]:
        assert not f.manual_kernel_registration

        if self.target is Target.REGISTRATION and not self.selector.is_native_function_selected(f):
            return None

        # TODO: Now, there is something interesting going on here.  In the code below,
        # we generate CompositeExplicitAutograd implementations of functional and inplace
        # based on the out implementation.  But in fact, out is definable by
        # functional too (just not very efficiently), and this is honestly the
        # MORE likely situation for a backend implementor.  How do we pick?
        # Well, taking a page from Haskell type classes and default methods,
        # we could conceivably register a circular definition (out in terms
        # of functional, and functional in terms of out) and just require
        # someone to implement one or the other.  We'd have to do a little bit
        # of work to not register one of these "weak" definitions unless there
        # is a strong definition somewhere in the DAG!  So it's not implemented yet.
        if self.dispatch_key == DispatchKey.CompositeExplicitAutograd and f.func.kind() is SchemaKind.out:
            # Never generate a default implementation for out, that's what you
            # have to define as a backend implementor
            return None

        # Note [Direct dispatch bindings]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Signature of the non-dispatched function we'll expose in a header
        # (e.g., at::cpu::add).  We don't generate methods (TODO: do this
        # when CPUTensor class is a thing); nor do we generate fallback
        # bindings for manual_cpp_binding functions.
        cpp_sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)

        # Signature of the wrapper function we'll register to the dispatcher
        sig = NativeSignature(f.func, prefix="wrapper_")

        if self.target is Target.NAMESPACED_DECLARATION:
            result = f"TORCH_API {cpp_sig_group.signature.decl()};\n"
            if cpp_sig_group.faithful_signature is not None:
                result += f"TORCH_API {cpp_sig_group.faithful_signature.decl()};\n"
            return result

        elif self.target is Target.NAMESPACED_DEFINITION:
            def generate_defn(cpp_sig: CppSignature) -> str:
                return f"""
{cpp_sig.defn()} {{
return {sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
"""
            result = generate_defn(cpp_sig_group.signature)
            if cpp_sig_group.faithful_signature is not None:
                result += generate_defn(cpp_sig_group.faithful_signature)
            return result

        elif self.target is Target.ANONYMOUS_DEFINITION:

            k = f.func.kind()

            # Construct the body of the wrapper function with signature sig
            sig_body = []
            # We'll use context to keep track of any variables we've brought
            # into scope while generating code
            context: List[Union[Binding, Expr]] = list(sig.arguments())

            # Initialize the class corresponding to this structured
            # operator; feeding it the output argument(s) if it is known
            if self.dispatch_key is DispatchKey.Meta:
                class_name = f"structured_{meta.name(self.g)}_meta_{k.name}"
                parent_class = f"at::meta::{meta.name(self.g)}"
            elif self.dispatch_key is DispatchKey.CompositeExplicitAutograd:
                # TODO: dedup this branch
                class_name = f"structured_{meta.name(self.g)}_default_backend_{k.name}"
                parent_class = f"at::meta::{meta.name(self.g)}"
            else:
                class_name = f"structured_{self.g.out.dispatch[self.dispatch_key]}_{k.name}"
                parent_class = f"at::native::structured_{self.g.out.dispatch[self.dispatch_key]}"

            if is_cuda_dispatch_key(self.dispatch_key):
                device_check_args = itertools.chain(
                    f.func.arguments.out,
                    f.func.arguments.flat_positional
                )
                sig_body.append(RegisterDispatchKey.gen_device_check(f.device_check, list(device_check_args), sig.name()))

            if k is SchemaKind.functional:
                sig_body.append(f"{class_name} op;")
            elif k is SchemaKind.inplace:
                sig_body.append(f"{class_name} op(self);")
            elif k is SchemaKind.out:
                out_args_str = ', '.join(a.name for a in f.func.arguments.out)
                sig_body.append(f"{class_name} op({out_args_str});")

            # Translate the input native arguments into structured
            # arguments for the meta call
            meta_exprs = ', '.join(
                e.expr for e in translate(
                    context,
                    structured.meta_arguments(self.g),
                    method=False
                )
            )
            sig_body.append(f"op.meta({meta_exprs});")

            # After running meta, op.outputs_ is guaranteed to be valid;
            # add it to the context
            out_args = structured.out_arguments(self.g)
            for i, out_arg in enumerate(out_args):
                assert ConstRefCType(BaseCType(tensorT)) == out_arg.nctype.type
                context.append(Expr(
                    expr=f"op.outputs_[{i}]",
                    # TODO: Stop hardcoding that the output type is a Tensor.  Note
                    # that for the codegen here this is fine because outputs_ is
                    # hardcoded to be tensor already
                    type=NamedCType(out_arg.nctype.name, MutRefCType(BaseCType(tensorT)))
                ))

            # With the expanded context, do the impl call (if not a meta
            # function)
            if self.dispatch_key == DispatchKey.CompositeExplicitAutograd:
                # TODO: https://github.com/pytorch/pytorch/issues/53023
                out_sig_group = CppSignatureGroup.from_native_function(
                    self.g.out, method=False, fallback_binding=f.manual_cpp_binding)
                out_sig = out_sig_group.most_faithful_signature()
                api_name = out_sig.name()
                out_exprs = ', '.join(
                    e.expr for e in translate(
                        context,
                        out_sig.arguments(),
                        method=False
                    )
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
            elif self.dispatch_key != DispatchKey.Meta:
                impl_exprs = ', '.join(
                    e.expr for e in translate(
                        context,
                        structured.impl_arguments(self.g),
                        method=False
                    )
                )
                sig_body.append(f"op.impl({impl_exprs});")

            # Destructively return the final tensors
            # TODO: Do this in translate instead
            if k is SchemaKind.functional:
                if len(f.func.returns) == 1:
                    ret_expr = "std::move(op.outputs_[0])"  # small optimization
                else:
                    moved = ', '.join(f"std::move(op.outputs_[{i}])" for i in range(len(f.func.returns)))
                    ret_expr = f"std::make_tuple({moved})"
            elif k is SchemaKind.inplace:
                ret_expr = "self"
            elif k is SchemaKind.out:
                if len(f.func.returns) == 1:
                    ret_expr = f.func.arguments.out[0].name
                else:
                    refs = ', '.join(a.name for a in f.func.arguments.out)
                    ret_expr = f"std::forward_as_tuple({refs})"
            sig_body.append(f"return {ret_expr};")

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
