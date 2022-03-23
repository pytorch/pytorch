from tools.codegen.api import cpp
from tools.codegen.api.types import (
    DispatcherSignature, Binding, FunctionalizationLambda, ViewInverseSignature,
    NativeSignature, Expr
)
from tools.codegen.api.translate import translate
from tools.codegen.context import (
    with_native_function, with_native_function_and_selector_and_index,
    with_native_function_and
)
from tools.codegen.model import (
    Argument, NativeFunction, SchemaKind, BackendIndex,
    Tag, FunctionSchema, SelfArgument, TensorOptionsArguments, BaseType, BaseTy,
    NativeFunctionsViewGroup,
)
from tools.codegen.selective_build.selector import SelectiveBuilder
from typing import List, Optional, Union, Tuple


# Generates the body of the default composite C++ kernel for a {view}_copy NativeFunction
# See Note [view_copy NativeFunctions]
@with_native_function
def gen_composite_view_copy_kernel(g: NativeFunctionsViewGroup) -> Optional[str]:

    if g.view_copy is None:
        return None
    docstr = f'''
add_docstr_all("{str(g.view_copy.func.name.name.base)}",
               r"""
Performs the same operation as :func:`torch.Tensor.{str(g.view.func.name.name.base)}`, but all output tensors
are freshly created instead of aliasing the input.
""")
'''
    print(docstr)
    view_copy_sig = NativeSignature(g.view_copy.func)
    view_sig = NativeSignature(g.view.func)

    view_tensor_name = view_copy_sig.arguments()[0].name
    view_clone_name = 'self_clone'
    view_api_name = g.view.func.name.unambiguous_name()

    # Set up the context for a translate() call. We don't want to include the first tensor argument
    # in the context, since we'll clone it and pass in the clone instead.
    context: List[Union[Binding, Expr]] = []
    view_copy_args = list(view_copy_sig.arguments())
    for b in view_copy_args[1:]:
        context.append(b)
    context.append(Expr(
        expr=view_clone_name,
        type=view_copy_args[0].nctype
    ))

    # The default generated composite kernel for {view}_copy() operators just clones
    # the input tensor, and runs the underlying view on the clone.
    return f"""
{view_copy_sig.defn()} {{
auto {view_clone_name} = {view_tensor_name}.clone();
return at::_ops::{view_api_name}::call({', '.join([e.expr for e in translate(context, view_sig.arguments())])});
}}
"""

def modifies_arguments(f: NativeFunction) -> bool:
    return f.func.kind() in [SchemaKind.inplace, SchemaKind.out]

# This function constructs the return statement for the kernels that contain mutations
# It mostly just needs to special case multi-output returns to wrap the result in a tuple
def return_str(f: NativeFunction) -> str:
    if len(f.func.arguments.out) != 0:
        if len(f.func.arguments.out) > 1:
            return_names = ', '.join(a.name for a in f.func.arguments.out)
            return f'return {DispatcherSignature.from_schema(f.func).returns_type().cpp_type()}({return_names});'
        else:
            return f'return {f.func.arguments.out[0].name}'
    if f.func.arguments.self_arg is not None:
        return f'return {f.func.arguments.self_arg.argument.name}'
    return ''

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
def unwrap_tensor_args(sig: DispatcherSignature, *, is_view_op: bool) -> Tuple[str, List[Binding]]:
    context: List[Binding] = []
    unwrapped_tensor_args: List[str] = []
    for arg in sig.arguments():
        if is_tensor_like(arg.argument):
            # for tensor inputs, we want to unwrap them before passing them into the redispatch calls.
            unwrapped_name = f'{arg.name}_'
            # For most ops, the functionalization needs to sync any pending updates on the input tensors
            # before calling the operator, since otherwise the operator will act on stale data.
            # For view ops though, we can continue to defer syncing until the tensor is used by
            # a non-view operator.
            maybe_sync_input = '' if is_view_op else f'at::functionalization::impl::sync({arg.name});'
            unwrapped_tensor_args.append(f"""
      {arg.nctype.remove_const_ref().cpp_type()} {unwrapped_name};
      if (at::functionalization::impl::isFunctionalTensor({arg.name})) {{
        {maybe_sync_input}
        {unwrapped_name} = at::functionalization::impl::from_functional_tensor({arg.name});
      }} else {{
        {unwrapped_name} = {arg.name};
      }}""")
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
    unwrap_tensor_args_str = '\n        '.join(unwrapped_tensor_args)
    return unwrap_tensor_args_str, context

# The functionalization codegen currently expects view op schemas to have this form:
# foo(Tensor(a), ...) -> Tensor(a) (e.g. transpose)
# foo(Tensor(a!), ...) -> Tensor(a!) (e.g. transpose_)
def assert_view_op_properties(func: FunctionSchema) -> None:
    def is_alias(a: Argument) -> bool:
        return a.annotation is not None

    args = func.arguments.flat_non_out
    # The first argument is a tensor with an alias semantics (annotations)
    assert len(args) > 0 and args[0].type == BaseType(BaseTy.Tensor), \
        f"""In the functionalization codegen, we expect the first argument of every view operator to be a tensor,
but found an argument of type {str(args[0].type)} for operator: {str(func.name)}."""
    # No other arguments have aliasing semantics
    assert is_alias(args[0]) and not any(is_alias(a) for a in args[1:]), \
        """In the functionalization codegen, we expect the first argument of every view operator to alias the output.
View operators with multiple aliasing inputs aren't supported yet. Found an operator that doesn't satisfy this constraint"""

# Generates the Functionalization kernel for:
# - ops that create aliases (e.g. transpose())
# - ops that are views AND mutations (e.g. transpose_())
@with_native_function_and
def emit_view_functionalization_body(
        f: NativeFunction,
        functional_op: NativeFunction
) -> str:
    # view op case
    assert f.is_view_op

    if f.tag is Tag.inplace_view:
        # This op is both an inplace op AND a view op.
        # See Note [Functionalization Pass - Inplace View Ops] for details.
        # I currently have the view meta call into the out-of-place variant of the view, to avoid
        # having to define an extra ~20 inplace {view}_inverse_ functions.
        # Most view ops don't have NativeFunctionGroup's both, because we don't define out= variants for view ops.
        # I'm assuming that every inplace-view op has a corresponding out-of-place view op,
        # with the same name but the trailing underscore removed.
        # This is currently asserted at parse time in gen.py (see error_check_native_functions).
        assert f.func.kind() is SchemaKind.inplace
        # Requirement: Every inplace_view op needs to have a corresponding functional view op, which we paired together beforehand.
        assert functional_op is not None
        api_name = functional_op.func.name.unambiguous_name()
        call_sig = DispatcherSignature.from_schema(functional_op.func)
    else:
        api_name = f.func.name.unambiguous_name()
        call_sig = DispatcherSignature.from_schema(f.func)

    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    assert_view_op_properties(f.func)
    view_tensor_name = dispatcher_sig.arguments()[0].name

    return_type = dispatcher_sig.returns_type().remove_const_ref().cpp_type()

    unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(dispatcher_sig, is_view_op=True)
    view_redispatch_args = [e.expr for e in translate(unwrapped_args_ctx, call_sig.arguments(), method=False)]

    forward_lambda = FunctionalizationLambda.from_func(f, functional_op=functional_op, is_reverse=False)
    reverse_lambda = FunctionalizationLambda.from_func(f, functional_op=functional_op, is_reverse=True)

    # The meta API call should use the same arguments, but convert all tensors to meta tensors first.
    meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
    meta_call_args = [e.expr for e in translate(meta_call_ctx, call_sig.arguments(), method=False)]

    if f.tag is Tag.inplace_view:
        # See Note [Functionalization Pass - Inplace View Ops] for more details
        return f"""
    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
        {forward_lambda.decl()} {{
          return {forward_lambda.inner_call()}
        }},
        {reverse_lambda.decl()} {{
          return {reverse_lambda.inner_call()}
        }}
      );
      at::functionalization::impl::mutate_view_meta({view_tensor_name}, view_meta);
      {unwrap_tensor_args_str}
      {return_type} reference_tensor_output;
      {{
        at::AutoDispatchSkipFunctionalize guard;
        {meta_conversion_str}
        reference_tensor_output = at::_ops::{api_name}::call({', '.join(meta_call_args)});
      }}
      // See  Note [Propagating strides in the functionalization pass]
      at::functionalization::impl::set_sizes_strides_offset({view_tensor_name}, reference_tensor_output);
      return {view_tensor_name};
    }}
"""

    else:
        return f"""
    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
      {unwrap_tensor_args_str}
      {return_type} tmp_output;
      {return_type} reference_tensor_output;
      {{
        at::AutoDispatchSkipFunctionalize guard;
        {meta_conversion_str}
        reference_tensor_output = at::_ops::{api_name}::call({', '.join(meta_call_args)});
        tmp_output = at::_ops::{api_name}::call({', '.join(view_redispatch_args)});
        // I'm fusing the [alias removal], [mutation removal], [add views back] passes together.
        // Later, we'll want to turn them into separate passes (since e.g. vulkan only cares about alias removal).
      }}
      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
        {forward_lambda.decl()} {{
          return {forward_lambda.inner_call()}
        }},
        {reverse_lambda.decl()} {{
          return {reverse_lambda.inner_call()}
        }}
      );
      auto out = at::functionalization::impl::create_functional_tensor_with_view_meta(tmp_output, {view_tensor_name}, view_meta);
      // See  Note [Propagating strides in the functionalization pass]
      at::functionalization::impl::set_sizes_strides_offset(out, reference_tensor_output);
      return out;
    }}
"""

# Generates the Functionalization kernel for inplace ops
@with_native_function_and
def emit_inplace_functionalization_body(
        f: NativeFunction,
        functional_op: Optional[NativeFunction]
) -> str:
    # mutation case
    assert(modifies_arguments(f))

    dispatcher_sig = DispatcherSignature.from_schema(f.func)

    return_type = dispatcher_sig.returns_type().remove_const_ref().cpp_type()

    unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(dispatcher_sig, is_view_op=False)

    maybe_return = '' if len(f.func.returns) == 0 else 'return '

    # Note [functionalizating copy_() and not preserving strides]
    # copy_() can't be functionalized, since there doesn't exist an out-of-place variant.
    # We could add one, but that would be sub-optimal for functorch: copy() would need to allocate a fresh tensor.
    # This may seem like a large hack for one optimization, but copy_() is one of the most common inplace operators.
    # Instead, we can replace `self.copy_(src)` with `src.to(self).expand_as(self)`.
    # This maintains the exact same semantics, EXCEPT that we don't preserve the strides from `self`.
    # This seems like a reasonable tradeoff, for a few reasons:
    # - mutation removal is only used by functorch, and not by Vulkan or XLA. Functorch already doesn't preserve strides.
    # - There are actually a few other places where the functionalization pass currently doesn't support strides:
    #   calls to slice/diagonal_scatter don't currently preserve the strides of their inputs (but maybe we should fix this).
    if str(f.func.name) == 'copy_':
        functional_call_str = """\
            auto tmp_intermediate = at::_ops::to_other::call(src_, self_, non_blocking, false, c10::nullopt);
            tmp_output = at::_ops::expand_as::call(tmp_intermediate, self_);"""
    elif functional_op is None:
        # We can't functionalize this inplace op, since we don't know what the corresponding functional op is.
        inplace_exprs = [e.expr for e in translate(unwrapped_args_ctx, dispatcher_sig.arguments(), method=False)]
        warn_str = "Note: the functionalization pass encountered an operator ({str(f.func.name)}) that it could not functionalize, \
because it couldn't find an out-of-place equivalent of the operator to call. \
Instead, it's calling the inplace/view operator directly. \
If this causes problems in your program, consider upstreaming the out-of-place op to PyTorch."

        return f"""
    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
      if (c10::impl::tls_local_dispatch_key_set().included_.has(c10::DispatchKey::Functionalize)) {{
          TORCH_WARN("{warn_str}");
      }}
      {unwrap_tensor_args_str}
      at::AutoDispatchSkipFunctionalize guard;
      // Redispatch as normally otherwise, since XLA has its own lowerings for special inplace ops.
      {maybe_return}at::_ops::{f.func.name.unambiguous_name()}::call({', '.join(inplace_exprs)});
    }}
"""
    else:
        # call the out-of-place variant of the op
        functional_sig = DispatcherSignature.from_schema(functional_op.func)
        functional_exprs = [e.expr for e in translate(unwrapped_args_ctx, functional_sig.arguments(), method=False)]
        functional_call_str = \
            f"tmp_output = at::_ops::{functional_op.func.name.unambiguous_name()}::call({', '.join(functional_exprs)});"

    mutable_input_post_processing = '\n'.join([
        f"""
      TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor({a.name}));
      auto {a.name}_functional = at::functionalization::impl::unsafeGetFunctionalWrapper({a.name});
      {a.name}_functional->replace_(tmp_output);
      {a.name}_functional->commit_update();"""
        for a in f.func.arguments.flat_non_out
        if a.annotation and a.annotation.is_write and a.type.is_tensor_like()])

    return f"""
    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
      {unwrap_tensor_args_str}
      {return_type} tmp_output;
      {{
          at::AutoDispatchSkipFunctionalize guard;
          // The functionalization pass explicitly doesn't pass out= parameters to the redispatch
          {functional_call_str}
      }}
      {mutable_input_post_processing}
      {return_str(f)};
    }}
"""


# The below functions generate RegisterFunctionalization.cpp
# These files provide the kernels that run the functionalization pass, which can be opted into
# per backend (e.g. XLA or Vulkan), or as a composable transform (functionalize() in functorch).

# TODO: later, I'll generate separate kernels for "Functionalize", and "AddBackViews"
# That will probably be an enum, that this function will take in to know which ops get kernels.
def needs_functionalization(
    selector: SelectiveBuilder,
    g: Union[NativeFunction, NativeFunctionsViewGroup],
) -> bool:
    # Don't generate kernels in mobile build
    if not selector.include_all_operators:
        return False
    # Every view op needs *some* handling in functionalization:
    # - non-composite view ops get a generated kernel
    # - composite view ops get a generated registration (that directly registers the composite kernel)
    if isinstance(g, NativeFunctionsViewGroup):
        return True
    # For non-view ops, only inplace operators get functionalized
    assert isinstance(g, NativeFunction)
    return modifies_arguments(g)


# See Note [Functionalization Pass: View Inverses].
def gen_functionalization_view_inverse_declaration(selector: SelectiveBuilder, g: NativeFunctionsViewGroup) -> Optional[str]:
    # For every (non-composite) view op, we need a corresponding "inverse view" function.
    # This generates the declarations so we get a good compiler error when someone adds a new view.
    @with_native_function
    def emit_decl_helper(g: NativeFunctionsViewGroup) -> Optional[str]:
        if g.view.has_composite_implicit_autograd_kernel:
            return None
        view_inverse_sig = ViewInverseSignature(g.view)
        return view_inverse_sig.decl()

    return emit_decl_helper(g)


@with_native_function_and_selector_and_index
def gen_functionalization_registration(
    selector: SelectiveBuilder,
    g: Union[NativeFunction, NativeFunctionsViewGroup],
    composite_implicit_autograd_index: BackendIndex
) -> List[str]:
    def emit_registration_helper(f: NativeFunction, *, is_view: bool) -> str:
        if is_view and f.has_composite_implicit_autograd_kernel:
            metadata = composite_implicit_autograd_index.get_kernel(f)
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
            # non-composite view ops (and inplace ops) get a normal registration.
            registration_str = f'TORCH_FN(functionalization::{wrapper_name(f.func)})'
        return f'm.impl("{f.func.name}", {registration_str});'

    # Note: for now, this logic is meant to avoid registering functionalization kernels for mobile.
    # At some point, Vulkan we'll want to use functionalization and we'll need to change this.
    if not needs_functionalization(selector, g):
        return []

    if isinstance(g, NativeFunctionsViewGroup):
        view_str = [emit_registration_helper(g.view, is_view=True)]
        if g.view_inplace is not None:
            view_str.append(emit_registration_helper(g.view_inplace, is_view=True))
        return view_str
    else:
        f = g
        return [emit_registration_helper(f, is_view=False)]

def gen_functionalization_definition(
    selector: SelectiveBuilder,
    g: Union[NativeFunction, NativeFunctionsViewGroup],
    functional_op: Optional[NativeFunction]
) -> List[str]:
    if not needs_functionalization(selector, g):
        return []
    if isinstance(g, NativeFunctionsViewGroup):
        view_defs = []
        if not g.view.has_composite_implicit_autograd_kernel:
            # For now I'm just asserting this - later, the view functionalization kernels
            # should be updated to redispatch to view_copy().
            assert g.view_copy is not None
            view_defs.append(emit_view_functionalization_body(g.view, g.view))
        if g.view_inplace is not None and not g.view_inplace.has_composite_implicit_autograd_kernel:
            view_defs.append(emit_view_functionalization_body(g.view_inplace, g.view))
        return view_defs
    else:
        f = g
        assert modifies_arguments(f)
        return [emit_inplace_functionalization_body(f, functional_op)]
