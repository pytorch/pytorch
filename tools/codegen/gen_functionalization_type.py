from tools.codegen.api import cpp
from tools.codegen.api import dispatcher
from tools.codegen.api.types import DispatcherSignature, CppSignatureGroup
from tools.codegen.api.translate import translate
from tools.codegen.context import method_with_native_function
from tools.codegen.model import (
    Argument, NativeFunction, NativeFunctionsGroup, Return, SchemaKind, BackendIndex,
    DispatchKey, Variant, Tag, BaseType, BaseTy
)
from typing import List, Optional, Union, Tuple, Dict
from dataclasses import dataclass
from typing_extensions import Literal
from tools.codegen.utils import mapMaybe, Target, assert_never

from tools.autograd.gen_inplace_or_view_type import (
    gen_formals, modifies_arguments
)
from tools.autograd.gen_trace_type import (
    type_wrapper_name
)

def is_view_op(f: NativeFunction) -> bool:
    rets = f.func.returns
    is_non_mutating_view = len(rets) > 0 and any(r.annotation is not None and not r.annotation.is_write for r in rets)
    is_inplace_view = f.tag is not None and f.tag is Tag.inplace_view
    return is_non_mutating_view or is_inplace_view

def return_names_str(f: NativeFunction) -> str:
    if len(f.func.arguments.out) != 0:
        if len(f.func.arguments.out) > 1:
            return_names = ', '.join(a.name for a in f.func.arguments.out)
            return f'{DispatcherSignature.from_schema(f.func).returns_type().cpp_type()}({return_names});'
        else:
            return f.func.arguments.out[0].name
    if f.func.arguments.self_arg is not None:
        return f.func.arguments.self_arg.argument.name
    raise AssertionError("Unable to handle functionalization for op={str(f.func.name)}")

def is_multi_output(rets: Tuple[Return, ...]) -> bool:
    return len(rets) > 1 or rets[0].type.is_list_like() is not None

def arg_name(a: Argument) -> str:
    return f'{a.name}_' if a.type.is_tensor_like() else a.name

def unwrap_tensor_args_str(f: NativeFunction) -> str:
    return '\n      '.join(
        f'auto {arg_name(arg)} = at::functionalization::impl::from_functional_tensor({arg.name});'
        for arg in f.func.arguments.flat_all if arg.type.is_tensor_like())

# Generates the Functionalization kernel for:
# - ops that create aliases (e.g. transpose())
# - ops that are views AND mutations (e.g. transpose_())
def emit_view_functionalization_body(f: NativeFunction, g: Optional[NativeFunctionsGroup], native_api_name: str) -> str:
    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_exprs = dispatcher_sig.exprs()

    keyset = 'ks & c10::after_func_keyset'
    return_type = dispatcher_sig.returns_type().remove_const_ref().cpp_type()

    api_name = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=f.manual_cpp_binding).most_faithful_signature().name()

    # view op case
    assert is_view_op(f)

    if f.tag is not None and f.tag is Tag.inplace_view:
        # This op is both an inplace op AND a view op.
        # See Note [Functionalization Pass - Inplace View Ops] for details.
        # I currently have the view meta call into the out-of-place variant of the view, to avoid
        # having to define an extra ~20 inplace {view}_inverse_ functions.
        # Most view ops don't have NativeFunctionGroup's both, because we don't define out= variants for view ops.
        # For now, I'm dangerously assuming that every inplace-view op has a corresponding out-of-place view op,
        # with the same name but the trailing underscore removed.
        assert f.func.kind() is SchemaKind.inplace
        api_name = api_name[:-1]


    non_self_args = dispatcher.jit_arguments(f.func, skip_self=True)
    for a in non_self_args:
        # There currently aren't any views that take extra tensor args aside from self.
        # If we eventually have any, the codegen will need extra logic to support it - for example,
        # when we call the at::meta:: API, all tensor args need to be converted to the meta device.
        assert not a.type.is_tensor_like()

    non_self_bindings = [dispatcher.argument(a) for a in non_self_args]
    non_self_value_ctypes = [cpp.argument_to_value_type(a) for a in non_self_args]
    non_self_capture_exprs = translate(non_self_bindings, non_self_value_ctypes, method=False)
    captured_view_args_str = ', '.join(f'{arg.name} = {val.expr}' for arg, val in zip(non_self_args, non_self_capture_exprs))

    multi_output = is_multi_output(f.func.returns)

    forward_lambda_decls = 'const at::Tensor& base, int64_t mutated_view_idx'
    reverse_lambda_decls = 'const at::Tensor& base, const at::Tensor& mutated_view, int64_t mutated_view_idx'

    if Variant.method in f.variants:
        forward_lambda_call_str = f'base.{api_name}({", ".join(a.name for a in non_self_args)})'
    else:
        forward_lambda_call_str = f'at::{api_name}({", ".join(["base"] + [a.name for a in non_self_args])})'
    reverse_lambda_args = ['base', 'mutated_view']

    if is_multi_output(f.func.returns):
        reverse_lambda_args.append('mutated_view_idx')
        maybe_idx = '[mutated_view_idx]'
    else:
        maybe_idx = ''

    reverse_lambda_args_str = ', '.join(reverse_lambda_args + [a.name for a in non_self_args])

    meta_call_args = [f'{a.name}_.to(kMeta)' if a.type.is_tensor_like() else a.name for a in f.func.arguments.flat_all]
    if Variant.method in f.variants:
        meta_call_str = f'{meta_call_args[0]}.{api_name}({", ".join(meta_call_args[1:])})'
    else:
        meta_call_str = f'at::{api_name}({", ".join(meta_call_args)})'

    view_meta_str = f"""
      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
        [{captured_view_args_str}]({forward_lambda_decls}) -> at::Tensor {{
          return {forward_lambda_call_str}{maybe_idx};
        }},
        [{captured_view_args_str}]({reverse_lambda_decls}) -> at::Tensor {{
          return at::functionalization::impl::{api_name}_inverse({reverse_lambda_args_str});
        }}
      );"""

    if f.tag is not None and f.tag is Tag.inplace_view:
        # See Note [Functionalization Pass - Inplace View Ops] for more details
        return f"""
      {view_meta_str}
      at::functionalization::impl::mutate_view_meta(self, view_meta);
      {unwrap_tensor_args_str(f)}
      {return_type} reference_tensor_output;
      {{
        at::AutoDispatchSkipFunctionalize guard;
        reference_tensor_output = {meta_call_str};
      }}
      // See  Note [Propagating strides in the functionalization pass]
      at::functionalization::impl::set_strides(self, reference_tensor_output);
      return self;
"""

    else:
        return f"""
      {unwrap_tensor_args_str(f)}
      {return_type} tmp_output;
      {return_type} reference_tensor_output;
      {{
        at::AutoDispatchSkipFunctionalize guard;
        reference_tensor_output = {meta_call_str};
        tmp_output = at::redispatch::{api_name}({', '.join([keyset] + [arg_name(a) for a in f.func.arguments.flat_all])});
        // I'm fusing the [alias removal], [mutation removal], [add views back] passes together.
        // Later, we'll want to turn them into separate passes (since e.g. vulkan only cares about alias removal).
      }}
      {view_meta_str}
      {return_type} out = at::functionalization::impl::create_functional_tensor_with_view_meta(tmp_output, self, view_meta);
      // See  Note [Propagating strides in the functionalization pass]
      at::functionalization::impl::set_strides(out, reference_tensor_output);
      return out;
"""

# Generates the Functionalization kernel for inplace ops
def emit_inplace_functionalization_body(f: NativeFunction, g: Optional[NativeFunctionsGroup]) -> str:
    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_exprs = dispatcher_sig.exprs()

    keyset = 'ks & c10::after_func_keyset'
    return_type = dispatcher_sig.returns_type().remove_const_ref().cpp_type()

    api_name = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=f.manual_cpp_binding).most_faithful_signature().name()

    # mutation case
    assert(modifies_arguments(f))

    maybe_return = '' if len(f.func.returns) == 0 else 'return '
    sync_tensor_args = '\n      '.join(mapMaybe(
        lambda arg: f'at::functionalization::impl::sync({arg.name});'
                    if arg.type.is_tensor_like() else None,
        f.func.arguments.flat_all))

    if g is None:
        # We can't functionalize this inplace op, since we don't know what the corresponding functional op is.
        cpp_sig = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding)
        warn_str = "Note: the functionalization pass encountered an operator ({}) that it could not functionalize, \
because it couldn't find an out-of-place equivalent of the operator to call. \
Instead, it's calling the inplace/view operator directly. \
If this causes problems in your program, consider upstreaming the out-of-place op to PyTorch.".format(str(f.func.name))

        return f"""
      if (c10::impl::tls_local_dispatch_key_set().included_.has(c10::DispatchKey::Functionalize)) {{
          TORCH_WARN("{warn_str}");
      }}
      {sync_tensor_args}
      {unwrap_tensor_args_str(f)}
      at::AutoDispatchSkipFunctionalize guard;
      // Redispatch as normally otherwise, since XLA has its own lowerings for special inplace ops.
      {maybe_return}at::redispatch::{api_name}({', '.join([keyset] + [arg_name(a) for a in f.func.arguments.flat_all])});
"""
    # call the out-of-place variant of the op
    functional_api_name = CppSignatureGroup.from_native_function(
        g.functional, method=False, fallback_binding=f.manual_cpp_binding).most_faithful_signature().name()

    mutable_input_post_processing = '\n'.join([
        f"""
      {a.name}.replace_(tmp_output);
      at::functionalization::impl::commit_update({a.name});"""
        for a in f.func.arguments.flat_non_out
        if a.annotation and a.annotation.is_write and a.type.is_tensor_like()])

    return f"""
      {sync_tensor_args}
      {unwrap_tensor_args_str(f)}
      {return_type} tmp_output;
      {{
          at::AutoDispatchSkipFunctionalize guard;
          // The functionalization pass explicitly doesn't pass out= parameters to the redispatch
          tmp_output = at::redispatch::{functional_api_name}(
            {', '.join([keyset] + [arg_name(a) for a in f.func.arguments.flat_non_out])});
      }}
      {mutable_input_post_processing}
      {maybe_return}{return_names_str(f)};"""



def emit_registration(f: NativeFunction, native_api_name: Optional[str]) -> str:
    if f.has_composite_implicit_autograd_kernel:
        assert native_api_name is not None, f"Found a view op ({f.func.name}) with no cpu or composite kernel." \
            f" The functionalization pass codegen needs to know the at::native::{f.func.name} kernel name of every view op," \
            f" which is currently does by looking up the cpu/composite kernel kernel name of the op. If you encountered a view op" \
            f" without a cpu/composite kernel, you'll need to change this logic"
        sig = DispatcherSignature.from_schema(f.func)
        # Note [Composite view ops in the functionalization pass]
        # We don't need to worry about implemententing functionalization kernels for views with
        # CompositeImplicitAutograd kernels, because we can just decompose them into their base operators.
        # We can't just opt the entire Functionalization dispatch key into the composite keyset though,
        # because we don't want to decompose non-view ops that are composite, like `at::ones`.
        registration_str = f'static_cast<{sig.ptr_type()}>(at::native::{native_api_name})'
    else :
        registration_str = f'TORCH_FN(functionalization::{type_wrapper_name(f)})'

    return f"""
m.impl("{f.func.name}",
       {registration_str}
);
"""

def emit_definition(f: NativeFunction, g: Optional[NativeFunctionsGroup], native_api_name: Optional[str]) -> str:
    # order is important here, ops that are both views and mutations should hit the view path.
    if is_view_op(f):
        assert native_api_name is not None, f"Found a view op ({f.func.name}) with no cpu or composite kernel." \
            f" The functionalization pass codegen needs to know the at::native::{f.func.name} kernel name of every view op," \
            f" which is currently does by looking up the cpu/composite kernel kernel name of the op. If you encountered a view op" \
            f" without a cpu/composite kernel, you'll need to change this logic"
        body_str = emit_view_functionalization_body(f, g, native_api_name)
    else:
        body_str = emit_inplace_functionalization_body(f, g)
    return f"""
{cpp.returns_type(f.func.returns).cpp_type()} {type_wrapper_name(f)}({gen_formals(f)}) {{
  {body_str}
}}
"""

def emit_declaration_for_noncomposite_views(f: NativeFunction) -> str:
    # For every view op, we need a corresponding "inverse view" function.
    # This generates the declarations so we get a good compiler error when someone adds a new view.
    for r in f.func.returns:
        assert r.type.is_tensor_like()

    name = dispatcher.name(f.func)
    non_self_args = [a.decl() for a in dispatcher.arguments(f.func, skip_self=True)]

    shared_args = ['const at::Tensor& base', 'const at::Tensor& mutated_view']
    if is_multi_output(f.func.returns):
        shared_args.append('int64_t mutated_view_idx')

    all_args = shared_args + non_self_args

    return f"Tensor {name}_inverse({', '.join(all_args)});"

def hopefully_get_native_api_name(f: NativeFunction, backend_indices: Dict[DispatchKey, BackendIndex]) -> Optional[str]:
    # Make a best-effort attempt to find the name of the native kernel.
    # This only matters for view ops.
    maybe_composite1_metadata = backend_indices[DispatchKey.CompositeImplicitAutograd].get_kernel(f)
    maybe_composite2_metadata = backend_indices[DispatchKey.CompositeExplicitAutograd].get_kernel(f)
    maybe_cpu_metadata = backend_indices[DispatchKey.CPU].get_kernel(f)
    # Sadly there are some sparse-only view ops.
    maybe_sparse_cpu_metadata = backend_indices[DispatchKey.SparseCPU].get_kernel(f)
    maybe_sparse_csr_cpu_metadata = backend_indices[DispatchKey.SparseCsrCPU].get_kernel(f)

    if maybe_composite1_metadata is not None:
        return maybe_composite1_metadata.kernel
    if maybe_composite2_metadata is not None:
        return maybe_composite2_metadata.kernel
    if maybe_cpu_metadata is not None:
        return maybe_cpu_metadata.kernel
    if maybe_sparse_cpu_metadata is not None:
        return maybe_sparse_cpu_metadata.kernel
    if maybe_sparse_csr_cpu_metadata is not None:
        return maybe_sparse_csr_cpu_metadata.kernel
    return None


# Generates RegisterFunctionalization.cpp
# These provide the kernels that run the functionalization pass, which can be opted into
# per backend (e.g. XLA or Vulkan), or as a composable transform (functionalize() in functorch).
@dataclass(frozen=True)
class Functionalize:
    # We don't really need a specific BackendIndex.
    # We just need to be able to call into the in-tree view ops, at::native::{view},
    # in order to compute accurate strides information.
    # Right now, I'm doing that by checking if each op has a cpu or composite kernel (which it currently does, for all view ops)
    backend_indices: Dict[DispatchKey, BackendIndex]
    target: Union[
        Literal[Target.REGISTRATION],
        Literal[Target.DEFINITION],
        Literal[Target.DECLARATION]
    ]

    @method_with_native_function
    def __call__(self, g: Union[NativeFunction, NativeFunctionsGroup]) -> List[str]:
        fs = [g] if isinstance(g, NativeFunction) else g.functions()
        group: Optional[NativeFunctionsGroup] = None if isinstance(g, NativeFunction) else g
        outputs = []
        for f in fs:
            native_api_name = hopefully_get_native_api_name(f, self.backend_indices)

            if not is_view_op(f) and not modifies_arguments(f):
                continue
            if self.target is Target.REGISTRATION:
                output = emit_registration(f, native_api_name)
            elif self.target is Target.DEFINITION:
                if is_view_op(f) and f.has_composite_implicit_autograd_kernel:
                    # See Note [Composite view ops in the functionalization pass]
                    continue
                output = emit_definition(f, group, native_api_name)
            elif self.target is Target.DECLARATION:
                if is_view_op(f) and not f.has_composite_implicit_autograd_kernel:
                    output = emit_declaration_for_noncomposite_views(f)
                else:
                    continue
            else:
                assert_never(self.target)

            outputs.append(output)
        return outputs
