from tools.codegen.api import cpp
from tools.codegen.api import dispatcher
from tools.codegen.api.types import (
    DispatcherSignature, CppSignatureGroup, Expr, Binding, NamedCType, ConstRefCType,
    BaseCType, tensorT, intT, CType
)
from tools.codegen.api.translate import translate
from tools.codegen.context import method_with_native_function
from tools.codegen.model import (
    Argument, NativeFunction, NativeFunctionsGroup, Return, SchemaKind, BackendIndex,
    Variant, Tag, FunctionSchema, SelfArgument, TensorOptionsArguments, BaseType, BaseTy,
    ListType
)
from typing import List, Optional, Union, Tuple, Sequence
from dataclasses import dataclass
from typing_extensions import Literal
from tools.codegen.utils import mapMaybe, Target, assert_never

from tools.autograd.gen_inplace_or_view_type import modifies_arguments

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

def wrapper_name(func: FunctionSchema) -> str:
    if func.name.overload_name:
        return f'{cpp.name(func)}_{func.name.overload_name}'
    else:
        return cpp.name(func)

def is_tensor_like(a: Union[Argument, TensorOptionsArguments, SelfArgument]) -> bool:
    return isinstance(a, SelfArgument) or (isinstance(a, Argument) and a.type.is_tensor_like())

# unwraps all tensor-like arguments, returning:
# (1) a string containing all of the logic that does the unwrapping
# (2) a context, to be used by translate(), with all of the relevant bindings.
def unwrap_tensor_args(sig: DispatcherSignature) -> Tuple[str, List[Binding]]:
    context: List[Binding] = []
    unwrapped_tensor_args: List[str] = []
    for arg in sig.arguments():
        if is_tensor_like(arg.argument):
            # for tensor inputs, we want to unwrap them before passing them into the redispatch calls.
            unwrapped_name = f'{arg.name}_'
            unwrapped_tensor_args.append(
                f'auto {unwrapped_name} = at::functionalization::impl::from_functional_tensor({arg.name});')
            context.append(arg.with_name(unwrapped_name))
        else:
            # for non-tensor inputs, we want to pass them directly into the redispatch calls.
            context.append(arg)
    unwrap_tensor_args_str = '\n      '.join(unwrapped_tensor_args)
    return unwrap_tensor_args_str, context

# converts  all tensor-like arguments to meta tensors, which are used to compute stride info. Returns:
# (1) a string containing all of the logic that does the conversions.
# (2) a context, to be used by translate(), with all of the relevant bindings.
def convert_to_meta_tensors(sig: DispatcherSignature) -> Tuple[str, List[Binding]]:
    context: List[Binding] = []
    unwrapped_tensor_args: List[str] = []
    for arg in sig.arguments():
        if is_tensor_like(arg.argument):
            # for tensor inputs, we want to unwrap them before passing them into the redispatch calls.
            # for tensor inputs, we want to unwrap them before passing them into the redispatch calls.
            a_ = arg.name
            unwrapped_name = f'{arg.name}_meta'
            unwrapped_tensor_args.append(
                f"auto {unwrapped_name} = at::native::empty_strided_meta({a_}.sizes(), {a_}.strides(), \
/*dtype=*/c10::make_optional({a_}.scalar_type()), /*layout=*/c10::make_optional({a_}.layout()), \
/*device=*/c10::make_optional(c10::Device(kMeta)), /*pin_memory=*/c10::nullopt);"
            )
            context.append(arg.with_name(unwrapped_name))
        else:
            # for non-tensor inputs, we want to pass them directly into the redispatch calls.
            context.append(arg)
    unwrap_tensor_args_str = '\n      '.join(unwrapped_tensor_args)
    return unwrap_tensor_args_str, context

# The function lambdas generated for each view op in the functionalization pass are of the form
# [captures](input_args) -> return_type {
#     return name(call_args);
# }
# This function returns the tuple of (input_args, name, return_type, call_args)
# One thing to note: for multi-output views, the forward lambda needs to call the op and index into the output.
# i.e. `return op(call_args)[idx]`
# In this case, the <call_args> binding list returned here includes the index argument,
# and the caller is responsible for parsing it out when generating the call-site.
def view_lambda_types(f: NativeFunction, *, reverse: bool) -> Tuple[List[Binding], str, CType, List[Binding]]:
    base_binding = Binding(
        name='base',
        nctype=NamedCType(name='base', type=ConstRefCType(BaseCType(tensorT))),
        argument=Argument(name='base', type=BaseType(BaseTy.Tensor), default=None, annotation=None),
        default=None)
    mutated_view_binding = Binding(
        name='mutated_view',
        nctype=NamedCType(name='mutated_view', type=ConstRefCType(BaseCType(tensorT))),
        argument=Argument(name='base', type=BaseType(BaseTy.Tensor), default=None, annotation=None),
        default=None)
    mutated_view_idx_binding = Binding(
        name='mutated_view_idx',
        nctype=NamedCType(name='mutated_view_idx', type=BaseCType(intT)),
        argument=Argument(name='base', type=BaseType(BaseTy.Tensor), default=None, annotation=None),
        default=None)

    non_self_args = dispatcher.jit_arguments(f.func, skip_self=True)
    for a in non_self_args:
        # There currently aren't any views that take extra tensor args aside from self.
        # If we eventually have any, the codegen will need extra logic to support it - for example,
        # when we call the at::meta:: API, all tensor args need to be converted to the meta device.
        assert not a.type.is_tensor_like()
    non_self_bindings = [dispatcher.argument(a) for a in non_self_args]

    # View ops always return a tensor or list of tensors.
    assert len(f.func.returns) > 0
    assert f.func.returns[0].type == BaseType(BaseTy.Tensor) or \
        (isinstance(f.func.returns[0].type, ListType) and f.func.returns[0].type.elem == BaseType(BaseTy.Tensor))

    return_type = BaseCType(tensorT)
    multi_out = is_multi_output(f.func.returns)
    name = dispatcher.name(f.func)
    if f.tag is not None and f.tag is Tag.inplace_view:
        # gen.py already asserts that ops tagged with inplace_view:
        # (a) end in a trailing underscore
        # (b) have a corresponding out-of-place op
        name = name[:-1]

    if reverse:
        name = f'{name}_inverse'
        input_args = [base_binding, mutated_view_binding, mutated_view_idx_binding]
        call_bindings = input_args + non_self_bindings if multi_out else input_args[:-1] + non_self_bindings
    else:
        input_args = [base_binding, mutated_view_idx_binding]
        # in forward multi-output case (e.g. split() forward), the index comes at the end
        # so we can parse it out appropriately.
        call_bindings = input_args[:-1] + non_self_bindings + [input_args[-1]] if multi_out else input_args[:-1] + non_self_bindings
    return input_args, name, return_type, call_bindings

# Generates the Functionalization kernel for:
# - ops that create aliases (e.g. transpose())
# - ops that are views AND mutations (e.g. transpose_())
def emit_view_functionalization_body(f: NativeFunction, g: Optional[NativeFunctionsGroup]) -> str:
    dispatcher_sig = DispatcherSignature.from_schema(f.func)

    keyset = 'dispatchKeySet & c10::after_func_keyset'
    return_type = dispatcher_sig.returns_type().remove_const_ref().cpp_type()

    # view op case
    assert is_view_op(f)

    unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(dispatcher_sig)
    view_redispatch_args = [keyset] + [e.expr for e in translate(unwrapped_args_ctx, dispatcher_sig.arguments(), method=False)]

    non_self_args = dispatcher.jit_arguments(f.func, skip_self=True)
    non_self_bindings = [dispatcher.argument(a) for a in non_self_args]
    non_self_value_ctypes = [cpp.argument_to_value_type(a) for a in non_self_args]
    # The goals of this translate are all C++ value types (like vector<int>),
    # while the bindings in the context are reference types (like IntArrayRef).
    # The purpose of the translate is to get out expressions that convert the references to value types,
    # so we can store them in a lambda capture.
    non_self_capture_exprs = translate(non_self_bindings, non_self_value_ctypes, method=False)
    captured_view_args_str = ', '.join(f'{val.type.name} = {val.expr}' for val in non_self_capture_exprs)
    # All of the captured variables are now in scope inside of the lambda. Add them to the context.
    lambda_capture_ctx = [Expr(expr=str(ctype.name), type=ctype) for ctype in non_self_value_ctypes]

    forward_lambda_args, forward_name, forward_ret_type, forward_call_bindings = view_lambda_types(f, reverse=False)
    reverse_lambda_args, reverse_name, reverse_ret_type, reverse_call_bindings = view_lambda_types(f, reverse=True)

    forward_lambda_ctx: Sequence[Expr] = lambda_capture_ctx + [Expr(expr=b.name, type=b.nctype) for b in forward_lambda_args]
    reverse_lambda_ctx: Sequence[Expr] = lambda_capture_ctx + [Expr(expr=b.name, type=b.nctype) for b in reverse_lambda_args]

    forward_lambda_exprs = [e.expr for e in translate(forward_lambda_ctx, forward_call_bindings, method=False)]
    reverse_lambda_exprs = [e.expr for e in translate(reverse_lambda_ctx, reverse_call_bindings, method=False)]

    maybe_idx = ''
    if is_multi_output(f.func.returns):
        #  mutated_view_idx is only needed for lambdas of multi-output view ops, like split().
        # For the forward lambda, we just directly call the corresponding view op and index into the output,
        # which is why mutated_view_idx isn't actually an argument to the function call.
        # For the reverse lambda, we have a calling convention for every {view}_reverse function such that
        # it only accepts an additional mutated_view_idx parameter if it's a multi-output view op.
        maybe_idx = f'[{forward_lambda_exprs[-1]}]'
        forward_lambda_exprs = forward_lambda_exprs[:-1]

    if Variant.method in f.variants:
        forward_lambda_call_str = f'{forward_lambda_exprs[0]}.{forward_name}({", ".join(forward_lambda_exprs[1:])}){maybe_idx}'
    else:
        forward_lambda_call_str = f'at::{forward_name}({", ".join(forward_lambda_exprs)}){maybe_idx}'

    view_meta_str = f"""
      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
        [{captured_view_args_str}]({', '.join(a.decl() for a in forward_lambda_args)}) -> {forward_ret_type.cpp_type()} {{
          return {forward_lambda_call_str};
        }},
        [{captured_view_args_str}]({', '.join(a.decl() for a in reverse_lambda_args)}) -> {reverse_ret_type.cpp_type()} {{
          return at::functionalization::impl::{reverse_name}({", ".join(reverse_lambda_exprs)});
        }}
      );"""

    api_name = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=f.manual_cpp_binding).most_faithful_signature().name()
    if f.tag is not None and f.tag is Tag.inplace_view:
        # This op is both an inplace op AND a view op.
        # See Note [Functionalization Pass - Inplace View Ops] for details.
        # I currently have the view meta call into the out-of-place variant of the view, to avoid
        # having to define an extra ~20 inplace {view}_inverse_ functions.
        # Most view ops don't have NativeFunctionGroup's both, because we don't define out= variants for view ops.
        # I'm assuming that every inplace-view op has a corresponding out-of-place view op,
        # with the same name but the trailing underscore removed.
        # This is currently asserted at parse time in gen.py (see error_check_native_functions).
        assert f.func.kind() is SchemaKind.inplace
        api_name = api_name[:-1]

    # The meta API call should use the same arguments, but convert all tensors to meta tensors first.
    meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
    meta_call_args = [e.expr for e in translate(meta_call_ctx, dispatcher_sig.arguments(), method=False)]
    if Variant.method in f.variants:
        assert is_tensor_like(meta_call_ctx[0].argument)
        meta_call_str = f'{meta_call_ctx[0].name}.{api_name}({", ".join(meta_call_args[1:])})'
    else:
        meta_call_str = f'at::{api_name}({", ".join(meta_call_args)})'


    if f.tag is not None and f.tag is Tag.inplace_view:
        # See Note [Functionalization Pass - Inplace View Ops] for more details
        return f"""
      {view_meta_str}
      at::functionalization::impl::mutate_view_meta(self, view_meta);
      {unwrap_tensor_args_str}
      {meta_conversion_str}
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
      {unwrap_tensor_args_str}
      {meta_conversion_str}
      {return_type} tmp_output;
      {return_type} reference_tensor_output;
      {{
        at::AutoDispatchSkipFunctionalize guard;
        reference_tensor_output = {meta_call_str};
        tmp_output = at::redispatch::{api_name}({', '.join(view_redispatch_args)});
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
    # mutation case
    assert(modifies_arguments(f))

    dispatcher_sig = DispatcherSignature.from_schema(f.func)

    keyset = 'dispatchKeySet & c10::after_func_keyset'
    return_type = dispatcher_sig.returns_type().remove_const_ref().cpp_type()

    unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(dispatcher_sig)

    maybe_return = '' if len(f.func.returns) == 0 else 'return '
    sync_tensor_args = '\n      '.join(mapMaybe(
        lambda arg: f'at::functionalization::impl::sync({arg.name});'
                    if arg.type.is_tensor_like() else None,
        f.func.arguments.flat_all))

    if g is None:
        # We can't functionalize this inplace op, since we don't know what the corresponding functional op is.
        api_name = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=f.manual_cpp_binding).most_faithful_signature().name()
        inplace_exprs = [keyset] + [e.expr for e in translate(unwrapped_args_ctx, dispatcher_sig.arguments(), method=False)]
        warn_str = "Note: the functionalization pass encountered an operator ({}) that it could not functionalize, \
because it couldn't find an out-of-place equivalent of the operator to call. \
Instead, it's calling the inplace/view operator directly. \
If this causes problems in your program, consider upstreaming the out-of-place op to PyTorch.".format(str(f.func.name))

        return f"""
      if (c10::impl::tls_local_dispatch_key_set().included_.has(c10::DispatchKey::Functionalize)) {{
          TORCH_WARN("{warn_str}");
      }}
      {sync_tensor_args}
      {unwrap_tensor_args_str}
      at::AutoDispatchSkipFunctionalize guard;
      // Redispatch as normally otherwise, since XLA has its own lowerings for special inplace ops.
      {maybe_return}at::redispatch::{api_name}({', '.join(inplace_exprs)});
"""
    # call the out-of-place variant of the op
    functional_sig = CppSignatureGroup.from_native_function(
        g.functional, method=False, fallback_binding=f.manual_cpp_binding).most_faithful_signature()
    functional_exprs = [keyset] + [e.expr for e in translate(unwrapped_args_ctx, functional_sig.arguments(), method=False)]

    mutable_input_post_processing = '\n'.join([
        f"""
      auto {a.name}_functional = at::functionalization::impl::unsafeGetFunctionalWrapper({a.name});
      {a.name}_functional->replace_(tmp_output);
      {a.name}_functional->commit_update();"""
        for a in f.func.arguments.flat_non_out
        if a.annotation and a.annotation.is_write and a.type.is_tensor_like()])

    return f"""
      {sync_tensor_args}
      {unwrap_tensor_args_str}
      {return_type} tmp_output;
      {{
          at::AutoDispatchSkipFunctionalize guard;
          // The functionalization pass explicitly doesn't pass out= parameters to the redispatch
          tmp_output = at::redispatch::{functional_sig.name()}({', '.join(functional_exprs)});
      }}
      {mutable_input_post_processing}
      {maybe_return}{return_names_str(f)};"""


def emit_declaration_for_noncomposite_views(f: NativeFunction) -> str:
    # For every view op, we need a corresponding "inverse view" function.
    # This generates the declarations so we get a good compiler error when someone adds a new view.
    _, name, return_type, reverse_call_bindings = view_lambda_types(f, reverse=True)
    return f"{return_type.cpp_type()} {name}({', '.join([b.decl() for b in reverse_call_bindings])});"


# Generates RegisterFunctionalization.cpp
# These provide the kernels that run the functionalization pass, which can be opted into
# per backend (e.g. XLA or Vulkan), or as a composable transform (functionalize() in functorch).
@dataclass(frozen=True)
class Functionalize:
    target: Union[
        Literal[Target.REGISTRATION],
        Literal[Target.DEFINITION]
    ]

    # Only used during registration.
    # We need to be able to register into the in-tree CompositeImplicitAutograd view ops, at::native::{view},
    # The index is used to figure out the name of the native kernel.
    composite_implicit_autograd_index: Optional[BackendIndex] = None

    @method_with_native_function
    def __call__(self, g: Union[NativeFunction, NativeFunctionsGroup]) -> List[str]:
        fs = [g] if isinstance(g, NativeFunction) else g.functions()
        group: Optional[NativeFunctionsGroup] = None if isinstance(g, NativeFunction) else g
        outputs = []
        for f in fs:
            if not is_view_op(f) and not modifies_arguments(f):
                continue
            if self.target is Target.REGISTRATION:
                output = self.emit_registration(f)
            elif self.target is Target.DEFINITION:
                if is_view_op(f) and f.has_composite_implicit_autograd_kernel:
                    # See Note [Composite view ops in the functionalization pass]
                    continue
                output = self.emit_definition(f, group)
            else:
                assert_never(self.target)

            outputs.append(output)
        return outputs

    def emit_registration(self, f: NativeFunction) -> str:
        if is_view_op(f) and f.has_composite_implicit_autograd_kernel:
            assert self.composite_implicit_autograd_index is not None
            metadata = self.composite_implicit_autograd_index.get_kernel(f)
            assert metadata is not None
            native_api_name = metadata.kernel
            sig = DispatcherSignature.from_schema(f.func)
            # Note [Composite view ops in the functionalization pass]
            # We don't need to worry about implemententing functionalization kernels for views with
            # CompositeImplicitAutograd kernels, because we can just decompose them into their base operators.
            # We can't just opt the entire Functionalization dispatch key into the composite keyset though,
            # because we don't want to decompose non-view ops that are composite, like `at::ones`.
            registration_str = f'static_cast<{sig.ptr_type()}>(at::native::{native_api_name})'
        else:
            registration_str = f'TORCH_FN(functionalization::{wrapper_name(f.func)})'

        return f"""
    m.impl("{f.func.name}",
        {registration_str}
    );
    """

    def emit_definition(self, f: NativeFunction, g: Optional[NativeFunctionsGroup]) -> str:
        # order is important here, ops that are both views and mutations should hit the view path.
        if is_view_op(f):
            body_str = emit_view_functionalization_body(f, g)
        else:
            # inplace op
            assert modifies_arguments(f)
            body_str = emit_inplace_functionalization_body(f, g)
        sig = DispatcherSignature.from_schema(f.func)
        return f"""
    {sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
    {body_str}
    }}
    """

# See Note [Functionalization Pass: View Inverses].
def gen_functionalization_view_inverse_declaration(g: Union[NativeFunction, NativeFunctionsGroup]) -> List[str]:
    fs = [g] if isinstance(g, NativeFunction) else g.functions()
    outputs = []
    for f in fs:
        if is_view_op(f) and not f.has_composite_implicit_autograd_kernel:
            output = emit_declaration_for_noncomposite_views(f)
            outputs.append(output)
    return outputs
