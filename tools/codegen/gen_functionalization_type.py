from tools.codegen.api import cpp
from tools.codegen.api.types import (
    DispatcherSignature, Binding, FunctionalizationLambda, ViewInverseSignature,
    NativeSignature
)
from tools.codegen.api.translate import translate
from tools.codegen.context import (
    with_native_function,
    with_native_function_and
)
from tools.codegen.model import (
    Argument, NativeFunction, SchemaKind, BackendIndex,
    Tag, FunctionSchema, SelfArgument, TensorOptionsArguments, BaseType, BaseTy,
    NativeFunctionsViewGroup, ListType
)
from tools.codegen.selective_build.selector import SelectiveBuilder
from tools.codegen.utils import Target

from typing import List, Optional, Union, Tuple
from enum import Enum

# This file contains codegen that relates to the functionalization pass.
# It includes:
# - gen_functionalization_definition
#     Generates dispatcher kernel definitions for the functionalization pass.
# - gen_functionalization_registration
#     Generates dispatcher kernel registrations for the functionalization pass.
# - gen_functionalization_view_inverse_declaration
#     Generates a declaration for an "inverse view", for every view op
#     that is needed in functionalization. We manually implement their definitions.
# - gen_composite_view_copy_kernel
#     Generates view_copy() composite kernels for all view_copy operators.


FunctionalizationTarget = Enum('Target', (
    'FUNCTIONALIZATION',
    'ADD_BACK_VIEWS',
))

# Generates the body of the default composite C++ kernel for a {view}_copy NativeFunction
# See Note [view_copy NativeFunctions]
@with_native_function
def gen_composite_view_copy_kernel(g: NativeFunctionsViewGroup) -> Optional[str]:

    if g.view_copy is None:
        return None
    # view_copy is a native signature, since we're generating an at::native:: kernel
    view_copy_sig = NativeSignature(g.view_copy.func)
    # view is a dispatcher signature, since we're calling into the at::_ops API
    view_sig = DispatcherSignature(g.view.func)

    view_api_name = g.view.func.name.unambiguous_name()
    exprs = ', '.join([e.expr for e in translate(view_copy_sig.arguments(), view_sig.arguments())])

    # view ops today always return either a Tensor or a list of Tensors
    assert len(g.view.func.returns) == 1
    assert g.view.func.returns[0].type == BaseType(BaseTy.Tensor) \
           or g.view.func.returns[0].type == ListType(BaseType(BaseTy.Tensor), None)

    if g.view.func.returns[0].type == BaseType(BaseTy.Tensor):
        return_cloned_output = '''\
  return output.clone();'''
    else:
        # If the return type is a list, we need to clone each tensor in the list.
        return_cloned_output = f'''\
  {view_copy_sig.returns_type().cpp_type()} out_clone;
  for (const auto i : c10::irange(output.size())) {{
    out_clone.push_back(output[i].clone());
  }}
  return out_clone;'''

    # The default generated composite kernel for {view}_copy() operators just clones
    # the input tensor, and runs the underlying view on the clone.
    return f"""
{view_copy_sig.defn()} {{
  auto output = at::_ops::{view_api_name}::call({exprs});
  {return_cloned_output}
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
    if f.func.arguments.self_arg is not None and len(f.func.returns) != 0:
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
        call_sig = DispatcherSignature.from_schema(functional_op.func)
    else:
        call_sig = DispatcherSignature.from_schema(f.func)

    # the "view_copy" op name that the functionalization kernels need to call
    api_name = functional_op.func.name.unambiguous_name()
    # Sometimes the functionalization pass needs to no-op (e.g. if it was passed non-functional tensors)
    # "no-op"ing in this context is just redispatching to the original op.
    noop_api_name = f.func.name.unambiguous_name()

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
      if (!at::functionalization::impl::isFunctionalTensor({view_tensor_name})) {{
        // functionalization is re-entrant, but will no-op if it wasn't passed a FunctionalTensorWrapper.
        {unwrap_tensor_args_str}
        at::AutoDispatchSkipFunctionalize guard;
        return at::_ops::{noop_api_name}::call({', '.join(view_redispatch_args)});
      }}
      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
        {forward_lambda.decl()} {{
          return {forward_lambda.inner_call()}
        }},
        {reverse_lambda.decl()} {{
          return {reverse_lambda.inner_call()}
        }}
      );
      at::functionalization::impl::mutate_view_meta({view_tensor_name}, view_meta);
      {return_type} reference_tensor_output;
      {{
        at::AutoDispatchSkipFunctionalize guard;
        {meta_conversion_str}
        reference_tensor_output = at::_ops::{noop_api_name}::call({', '.join(meta_call_args)});
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
      if (!at::functionalization::impl::isFunctionalTensor({view_tensor_name})) {{
        // functionalization is re-entrant, but will no-op if it wasn't passed a FunctionalTensorWrapper.
        at::AutoDispatchSkipFunctionalize guard;
        return at::_ops::{noop_api_name}::call({', '.join(view_redispatch_args)});
      }}
      {return_type} tmp_output;
      {return_type} reference_tensor_output;
      {{
        at::AutoDispatchSkipFunctionalize guard;
        {meta_conversion_str}
        reference_tensor_output = at::_ops::{noop_api_name}::call({', '.join(meta_call_args)});
        tmp_output = at::_ops::{api_name}::call({', '.join(view_redispatch_args)});
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

# Generates the Functionalization kernel for:
# - mutation ops (inplace and out= ops)
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

    mutated_names = [a.name for a in f.func.arguments.flat_all if a.type.is_tensor_like() and a.annotation is not None]
    non_mutated_names = [a.name for a in f.func.arguments.flat_all if a.type.is_tensor_like() and a.annotation is None]
    # all mutable inputs must be functional tensors in order to participate in functionalization
    check_all_mutated_args_are_functional = ' && '.join(
        ['true'] + [f'at::functionalization::impl::isFunctionalTensor({a})' for a in mutated_names])
    check_any_non_mutated_args_are_functional = ' || '.join(
        ['false'] + [f'at::functionalization::impl::isFunctionalTensor({a})' for a in non_mutated_names])
    # These are used in the cases where we don't functionalize and redispatch to the inplace op
    # case 1: we hit an inplace op that doesn't have an out-of-place equivalent
    # case 2: we hit an inplace ops but our inputs are not functional tensors (in which case our kernel just no-ops)
    inplace_exprs = [e.expr for e in translate(unwrapped_args_ctx, dispatcher_sig.arguments(), method=False)]

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
            tmp_output = at::_ops::expand_copy::call(tmp_intermediate, self_.sizes(), false);"""
    elif functional_op is None:
        # We can't functionalize this inplace op, since we don't know what the corresponding functional op is.
        warn_str = f"""Note: the functionalization pass encountered an operator ({str(f.func.name)}) that it could not \
functionalize, because it couldn't find an out-of-place equivalent of the operator to call. \
Instead, it's calling the inplace/view operator directly. \
If this causes problems in your program, consider upstreaming the out-of-place op to PyTorch."""

        return f"""
    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
      if (c10::impl::tls_local_dispatch_key_set().included_.has(c10::DispatchKey::Functionalize)) {{
          TORCH_WARN("{warn_str}");
      }}
      {unwrap_tensor_args_str}
      at::AutoDispatchSkipFunctionalize guard;
      // Redispatch as normally otherwise, since XLA has its own lowerings for special inplace ops.
      at::_ops::{f.func.name.unambiguous_name()}::call({', '.join(inplace_exprs)});
      {return_str(f)};
    }}
"""
    else:
        # call the out-of-place variant of the op
        functional_sig = DispatcherSignature.from_schema(functional_op.func)
        functional_exprs = [e.expr for e in translate(unwrapped_args_ctx, functional_sig.arguments(), method=False)]
        functional_call_str = \
            f"tmp_output = at::_ops::{functional_op.func.name.unambiguous_name()}::call({', '.join(functional_exprs)});"

    if f.func.is_out_fn():
        mutable_input_post_processing = '\n'.join([
            f"""
      auto {a.name}_functional = at::functionalization::impl::unsafeGetFunctionalWrapper({a.name});
      {a.name}_functional->replace_({'std::get<' + str(i) + '>(tmp_output)' if len(f.func.returns) > 1 else 'tmp_output'});
      {a.name}_functional->commit_update();"""
            for (i, a) in enumerate(f.func.arguments.out) if a.annotation and a.annotation.is_write and a.type.is_tensor_like()])
    else:
        mutable_input_post_processing = '\n'.join([
            f"""
      auto {a.name}_functional = at::functionalization::impl::unsafeGetFunctionalWrapper({a.name});
      {a.name}_functional->replace_(tmp_output);
      {a.name}_functional->commit_update();"""
            for a in f.func.arguments.flat_all
            if a.annotation and a.annotation.is_write and a.type.is_tensor_like()])

    return f"""
    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
      {unwrap_tensor_args_str}
      if (!({check_all_mutated_args_are_functional})) {{
        if (({check_any_non_mutated_args_are_functional})) {{
         // case 1: trying to mutate a non functional tensor with a functional tensor is an error
         TORCH_INTERNAL_ASSERT(false,
           "mutating a non-functional tensor with a functional tensor is not allowed.",
           " Please ensure that all of your inputs are wrapped inside of a functionalize() call.");
        }} else {{
         // case 2: arguments are not functional tensors, so we no-op and redispatch.
         at::AutoDispatchSkipFunctionalize guard;
         at::_ops::{f.func.name.unambiguous_name()}::call({', '.join(inplace_exprs)});
        {return_str(f)};
        }}
      }} else {{
        {return_type} tmp_output;
        {{
          at::AutoDispatchSkipFunctionalize guard;
          {functional_call_str}
        }}
        {mutable_input_post_processing}
        {return_str(f)};
      }}
    }}"""

# Generates the FunctionalizeAddBackViews kernels for:
# - view_copy ops (which get turned back into view ops)
def emit_view_copy_to_view_kernel(g: NativeFunctionsViewGroup) -> str:
    view_sig = DispatcherSignature(g.view.func)
    assert g.view_copy is not None
    view_copy_sig = DispatcherSignature(g.view_copy.func)

    view_api_name = g.view.func.name.unambiguous_name()
    exprs = ', '.join([e.expr for e in translate(view_copy_sig.arguments(), view_sig.arguments())])

    # The default generated composite kernel for {view}_copy() operators just clones
    # the input tensor, and runs the underlying view on the clone.
    return f"""
// re-apply views "underneath" the functionalization pass, by converting view_copy calls to view calls.
{view_copy_sig.defn(name=wrapper_name(g.view_copy.func), is_redispatching_fn=True)} {{
  return at::_ops::{view_api_name}::call({exprs});
}}
"""

# The below functions generate RegisterFunctionalization.cpp
# These files provide the kernels that run the functionalization pass, which can be opted into
# per backend (e.g. XLA or Vulkan), or as a composable transform (functionalize() in functorch).

def needs_functionalization(
    selector: SelectiveBuilder,
    g: Union[NativeFunction, NativeFunctionsViewGroup],
    target: Target,
    funcTarget: FunctionalizationTarget,
) -> bool:
    # Don't generate kernels in mobile build
    if not selector.include_all_operators:
        return False

    # Case 1: Functionalization kernels
    # all view + inplace ops get a functional kernel registration + definition
    if funcTarget == FunctionalizationTarget.FUNCTIONALIZATION:
        if isinstance(g, NativeFunctionsViewGroup):
            return True
        assert isinstance(g, NativeFunction)
        if modifies_arguments(g):
            return True
    else:
        # Case 2: "Add back view" kernels
        # only view_copy ops get a kernel definition + registration
        assert funcTarget == FunctionalizationTarget.ADD_BACK_VIEWS
        if isinstance(g, NativeFunctionsViewGroup):
            return True

    # Case 3: CompositeImplicitAutograd kernels
    # For both Functionalization and AddBackView, we *always* register the composite kernel
    # if there is one, so we decompose before hitting the functionalization pass.
    # (We don't generate a definition though).
    # If/when we separate "decomposition" into a separate dispatch key, we won't need to do this.
    f = g.view if isinstance(g, NativeFunctionsViewGroup) else g
    return f.has_composite_implicit_autograd_kernel and target == Target.REGISTRATION


# See Note [Functionalization Pass: View Inverses].
def gen_functionalization_view_inverse_declaration(selector: SelectiveBuilder, g: NativeFunctionsViewGroup) -> Optional[str]:
    # For every (non-composite) view op, we need a corresponding "inverse view" function.
    # This generates the declarations so we get a good compiler error when someone adds a new view.
    @with_native_function
    def emit_decl_helper(g: NativeFunctionsViewGroup) -> Optional[str]:
        if g.view.has_composite_implicit_autograd_kernel:
            return None
        view_copy_inverse_sig = ViewInverseSignature(g)
        return view_copy_inverse_sig.decl()

    return emit_decl_helper(g)


def gen_functionalization_registration(
    selector: SelectiveBuilder,
    g: Union[NativeFunction, NativeFunctionsViewGroup],
    composite_implicit_autograd_index: BackendIndex,
    funcTarget: FunctionalizationTarget,
) -> List[str]:
    @with_native_function
    def emit_registration_helper(f: NativeFunction) -> str:
        if f.has_composite_implicit_autograd_kernel:
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
            if funcTarget == FunctionalizationTarget.FUNCTIONALIZATION:
                registration_str = f'TORCH_FN(functionalization::{wrapper_name(f.func)})'
            else:
                assert funcTarget == FunctionalizationTarget.ADD_BACK_VIEWS
                registration_str = f'TORCH_FN(functionalization::addbackviews::{wrapper_name(f.func)})'
        return f'm.impl("{f.func.name}", {registration_str});'

    # Note: for now, this logic is meant to avoid registering functionalization kernels for mobile.
    # At some point, Vulkan we'll want to use functionalization and we'll need to change this.
    if not needs_functionalization(selector, g, Target.REGISTRATION, funcTarget):
        return []

    if isinstance(g, NativeFunctionsViewGroup):
        # we always want to register composite kernels
        if g.view.has_composite_implicit_autograd_kernel:
            view_str = [emit_registration_helper(g.view)]
            if g.view_inplace is not None:
                assert g.view_inplace.is_view_op
                # We don't currently have any composite view ops that have a non-composite
                # "inplace_view" variant, and implementing the logic it is a little annoying.
                assert g.view_inplace.has_composite_implicit_autograd_kernel
                view_str.append(emit_registration_helper(g.view_inplace))
            return view_str

        # functionalization needs to register kernels for view + view_inplace ops
        if funcTarget == FunctionalizationTarget.FUNCTIONALIZATION:
            view_str = [emit_registration_helper(g.view)]
            if g.view_inplace is not None:
                assert g.view_inplace.is_view_op
                view_str.append(emit_registration_helper(g.view_inplace))
            return view_str

        # while "add back view" kernels need to register kernels for view_copy ops
        assert funcTarget == FunctionalizationTarget.ADD_BACK_VIEWS
        assert g.view_copy is not None
        return [emit_registration_helper(g.view_copy)]
    else:
        f = g
        assert not f.is_view_op
        return [emit_registration_helper(f)]

def gen_functionalization_definition(
    selector: SelectiveBuilder,
    g: Union[NativeFunction, NativeFunctionsViewGroup],
    functional_op: Optional[NativeFunction],
    funcTarget: FunctionalizationTarget,
) -> List[str]:
    if not needs_functionalization(selector, g, Target.ANONYMOUS_DEFINITION, funcTarget):
        return []
    if funcTarget == FunctionalizationTarget.FUNCTIONALIZATION:
        if isinstance(g, NativeFunctionsViewGroup):
            # Case 1: emit view -> view_copy kernels for the functionalization pass
            view_defs = []
            if not g.view.has_composite_implicit_autograd_kernel:
                # invariant: NativeFunctionsViewGroup's always have a view_copy operator
                # if the view is not composite (implicit autograd)
                assert g.view_copy is not None
                view_defs.append(emit_view_functionalization_body(g.view, g.view_copy))
            if g.view_inplace is not None and not g.view_inplace.has_composite_implicit_autograd_kernel:
                assert g.view_copy is not None
                view_defs.append(emit_view_functionalization_body(g.view_inplace, g.view_copy))
            return view_defs
        else:
            # Case 2: emit inplace -> out-of-place kernels for the functionalization pass
            f = g
            assert modifies_arguments(f)
            return [emit_inplace_functionalization_body(f, functional_op)]

    # Case 3: emit view_copy -> view kernels, adding back views "underneath" the functionalization pass.
    # Only do it for non-composite views though, since those have already been decomposed.
    # (we register the decompositions directly, instead)
    assert funcTarget == FunctionalizationTarget.ADD_BACK_VIEWS
    if isinstance(g, NativeFunctionsViewGroup) and not g.view.has_composite_implicit_autograd_kernel:
        return [emit_view_copy_to_view_kernel(g)]
    return []
