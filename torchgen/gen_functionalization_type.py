from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    CType,
    DispatcherSignature,
    FunctionalizationLambda,
    iTensorListRefT,
    NativeSignature,
    OptionalCType,
    optionalSymIntArrayRefT,
    symIntArrayRefT,
    SymIntT,
    tensorListT,
    tensorT,
    VectorCType,
    ViewInverseSignature,
)
from torchgen.context import (
    method_with_native_function,
    native_function_manager,
    with_native_function,
    with_native_function_and,
)
from torchgen.model import (
    Argument,
    BackendIndex,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    Return,
    SchemaKind,
    SelfArgument,
    TensorOptionsArguments,
)
from torchgen.native_function_generation import (
    INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY,
    MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT,
    OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY,
)

from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import dataclass_repr


# Note: [Mutable Ops Not Using Functionalization]
# Ops in this list currently do not work with functionalization and should be fixed.
MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION = (
    OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY
    + MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT
    + INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY
    + [
        # It will be BC-breaking, but we should fix their schemas.
        # should be inplace?
        "record_stream",
        # See Note [resize_ in Functionalization]
        "resize_",
        "resize_as_",
        # This function is used as for testing purposes only.
        "_fill_mem_eff_dropout_mask_",
    ]
)

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


# Generates the body of the default composite C++ kernel for a {view}_copy NativeFunction
# See Note [view_copy NativeFunctions]
@dataclass(frozen=True)
class GenCompositeViewCopyKernel:
    backend_index: BackendIndex

    @method_with_native_function
    def __call__(self, g: NativeFunctionsViewGroup) -> Optional[str]:
        if g.view_copy is None:
            return None
        elif g.view_copy.func.name.name.base != f"{g.view.func.name.name}_copy":
            # If the view_copy doesn't match the standard naming scheme of <op>_copy,
            # assume it already exists and doesn't need to be generated.
            # Example: slice_inverse() with the copy variant named slice_scatter()
            # instead of slice_inverse_copy()
            return None

        metadata = self.backend_index.get_kernel(g.view_copy)
        assert metadata is not None

        # We can make view_copy work in more cases by using reshape()
        # when a normal view call would ordinarily fail.
        # This also makes LTC more efficient, because they don't need to include
        # clone() calls in their graph (which is normally needed by reshape).
        if str(g.view_copy.func.name) == "view_copy":
            assert metadata.kernel == "view_copy_symint"
            return """\
at::Tensor view_copy_symint(const at::Tensor & self, at::SymIntArrayRef size) {
  c10::SymDimVector shape = infer_size_dv(size, self.sym_numel());
  if (!at::detail::computeStride(self.sym_sizes(), self.sym_strides(), shape).has_value()) {
    return self.reshape_symint(size);
  } else {
    auto output = at::_ops::view::call(self, size);
    return output.clone(/*memory_format=*/at::MemoryFormat::Contiguous);
  }
}
"""
        # view_copy is a native signature, since we're generating an at::native:: kernel
        # Functionalization always operates on symints though
        view_copy_sig = NativeSignature(
            g.view_copy.func, symint=metadata.supports_symint()
        )

        # view is a dispatcher signature, since we're calling into the at::_ops API
        view_sig = DispatcherSignature(g.view.func)

        view_api_name = g.view.func.name.unambiguous_name()
        exprs = ", ".join(
            [e.expr for e in translate(view_copy_sig.arguments(), view_sig.arguments())]
        )

        # view ops today always return either a Tensor or a list of Tensors
        assert len(g.view.func.returns) == 1
        assert g.view.func.returns[0].type == BaseType(
            BaseTy.Tensor
        ) or g.view.func.returns[0].type == ListType(BaseType(BaseTy.Tensor), None)

        if g.view.func.returns[0].type == BaseType(BaseTy.Tensor):
            return_cloned_output = """\
  return output.clone(/*memory_format=*/at::MemoryFormat::Contiguous);"""
        else:
            # If the return type is a list, we need to clone each tensor in the list.
            return_cloned_output = f"""\
  {view_copy_sig.returns_type().cpp_type()} out_clone;
  for (const auto i : c10::irange(output.size())) {{
    out_clone.push_back(output[i].clone(/*memory_format=*/at::MemoryFormat::Contiguous));
  }}
  return out_clone;"""

        # The default generated composite kernel for {view}_copy() operators just clones
        # the input tensor, and runs the underlying view on the clone.
        return f"""
{view_copy_sig.defn(name=metadata.kernel)} {{
  auto output = at::_ops::{view_api_name}::call({exprs});
  {return_cloned_output}
}}
"""


def return_str(rets: Tuple[Return, ...], names: List[str]) -> str:
    assert len(rets) == len(names)
    if len(rets) == 0:
        return ""
    elif len(rets) == 1:
        return f"return {names[0]};"
    else:
        return f"return {dispatcher.returns_type(rets).cpp_type()}({', '.join(names)});"


def modifies_arguments(f: NativeFunction) -> bool:
    return any(
        a.annotation is not None and a.annotation.is_write
        for a in f.func.arguments.flat_all
    )


def wrapper_name(func: FunctionSchema) -> str:
    if func.name.overload_name:
        return f"{cpp.name(func)}_{func.name.overload_name}"
    else:
        return cpp.name(func)


def is_tensor_like(a: Union[Argument, TensorOptionsArguments, SelfArgument]) -> bool:
    return isinstance(a, SelfArgument) or (
        isinstance(a, Argument) and a.type.is_tensor_like()
    )


# We need to wrap / unwrap various arguments from the op in the functionalization kernels.
# Some op schemas include non-owning types though (like TensorList),
# and when we unwrap them we expect to get out an owning type!.
# We also return a lambda that tells you how to conver the non-owning type argument into the owning type.
def get_owning_type(t: CType) -> Tuple[CType, Callable[[str], str]]:
    if t == BaseCType(tensorListT):
        return VectorCType(BaseCType(tensorT)), lambda x: f"{x}.vec()"
    if t == BaseCType(iTensorListRefT):
        return VectorCType(BaseCType(tensorT)), lambda x: f"{{{x}.begin(), {x}.end()}}"
    # There are technically other non-owning types out there (like IntArrayRef),
    # but functionalization only actually cares about the ones involving tensors.
    return t, lambda x: x


# unwraps all tensor-like arguments, returning:
# (1) a string containing all of the logic that does the unwrapping
# (2) a context, to be used by translate(), with all of the relevant bindings.
def unwrap_tensor_args(
    sig: DispatcherSignature, *, is_view_op: bool
) -> Tuple[str, List[Binding]]:
    context: List[Binding] = []
    unwrapped_tensor_args: List[str] = []
    for arg in sig.arguments():
        if is_tensor_like(arg.argument):
            # for tensor inputs, we want to unwrap them before passing them into the redispatch calls.
            unwrapped_name = f"{arg.name}_"
            # For most ops, the functionalization needs to sync any pending updates on the input tensors
            # before calling the operator, since otherwise the operator will act on stale data.
            # For view ops though, we can continue to defer syncing until the tensor is used by
            # a non-view operator.
            maybe_sync_input = (
                "" if is_view_op else f"at::functionalization::impl::sync({arg.name});"
            )
            unwrapped_type, conversion_fn = get_owning_type(
                arg.nctype.remove_const_ref().type
            )
            unwrapped_tensor_args.append(
                f"""
      {unwrapped_type.cpp_type()} {unwrapped_name};
      if (at::functionalization::impl::isFunctionalTensor({arg.name})) {{
        {maybe_sync_input}
        {unwrapped_name} = at::functionalization::impl::from_functional_tensor({arg.name});
      }} else {{
        {unwrapped_name} = {conversion_fn(arg.name)};
      }}"""
            )
            context.append(arg.with_name(unwrapped_name))
        else:
            # for non-tensor inputs, we want to pass them directly into the redispatch calls.
            context.append(arg)
    unwrap_tensor_args_str = "\n      ".join(unwrapped_tensor_args)
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
            a_ = arg.name
            unwrapped_name = f"{arg.name}_meta"
            unwrapped_tensor_args.append(f"auto {unwrapped_name} = to_meta({a_});")
            context.append(arg.with_name(unwrapped_name))
        else:
            # for non-tensor inputs, we want to pass them directly into the redispatch calls.
            context.append(arg)
    unwrap_tensor_args_str = "\n        ".join(unwrapped_tensor_args)
    return unwrap_tensor_args_str, context


# The functionalization codegen currently expects view op schemas to have this form:
# foo(Tensor(a), ...) -> Tensor(a) (e.g. transpose)
# foo(Tensor(a!), ...) -> Tensor(a!) (e.g. transpose_)
def assert_view_op_properties(func: FunctionSchema) -> None:
    def is_alias(a: Argument) -> bool:
        return a.annotation is not None

    args = func.arguments.flat_non_out
    # The first argument is a tensor with an alias semantics (annotations)
    assert len(args) > 0 and args[0].type == BaseType(
        BaseTy.Tensor
    ), f"""In the functionalization codegen, we expect the first argument of every view operator to be a tensor,
but found an argument of type {str(args[0].type)} for operator: {str(func.name)}."""
    # No other arguments have aliasing semantics
    assert is_alias(args[0]) and not any(
        is_alias(a) for a in args[1:]
    ), """In the functionalization codegen, we expect the first argument of every view operator to alias the output.
View operators with multiple aliasing inputs aren't supported yet. Found an operator that doesn't satisfy this constraint"""


# One-liner expression for checking if an expression expr of type type has any
# symbolic values.
def emit_expr_has_symbolic_values(expr: str, type: CType) -> str:
    if type == BaseCType(SymIntT):
        return f"{expr}.is_symbolic()"

    if isinstance(type, OptionalCType):
        innerexpr = f"(*{expr})"
        return f"{expr}.has_value() ? {emit_expr_has_symbolic_values(innerexpr, type.elem)} : false"

    if type == BaseCType(optionalSymIntArrayRefT):
        return emit_expr_has_symbolic_values(
            expr, OptionalCType(BaseCType(symIntArrayRefT))
        )

    if type in (BaseCType(symIntArrayRefT), VectorCType(BaseCType(SymIntT))):
        argname = "arg"
        lambda_check = emit_expr_has_symbolic_values(argname, BaseCType(SymIntT))
        return (
            "std::any_of("
            f"{expr}.begin(), {expr}.end(), "
            f"[=](auto& {argname}) {{ return {lambda_check}; }})"
        )

    raise ValueError(
        "unsupported type for has_symbolic_values check. "
        "It should be a SymInt or a collection of those. "
        f"Got: {type.cpp_type()}"
    )


# Detects whether any of the SymInt arguments are, in fact, symbolic values.
# This is used in the constructor of ViewMeta.
def emit_has_symbolic_inputs(sig: DispatcherSignature) -> Tuple[str, str]:
    name = "has_symbolic_inputs"
    statements = [
        f"{name} = {name} | ({emit_expr_has_symbolic_values(binding.name, binding.nctype.type)});"
        for binding in sig.arguments()
        if (
            isinstance(binding.argument, Argument)
            and binding.argument.type.is_symint_like()
        )
    ]
    body = "\n      ".join(statements)
    return (
        name,
        f"""
      bool {name} = false;
      {body}""",
    )


# Generates the Functionalization kernel for:
# - ops that create aliases (e.g. transpose())
# - ops that are views AND mutations (e.g. transpose_())
def emit_view_functionalization_body(
    g: NativeFunctionsViewGroup, *, view_inplace: bool
) -> str:
    if view_inplace:
        # This op is both an inplace op AND a view op.
        # See Note [Functionalization Pass - Inplace View Ops] for details.
        # I currently have the view meta call into the out-of-place variant of the view, to avoid
        # having to define an extra ~20 inplace {view}_inverse_ functions.
        # Most view ops don't have NativeFunctionGroup's both, because we don't define out= variants for view ops.
        # I'm assuming that every inplace-view op has a corresponding out-of-place view op,
        # with the same name but the trailing underscore removed.
        # This is currently asserted at parse time in gen.py (see error_check_native_functions).
        assert g.view_inplace is not None
        f = g.view_inplace
    else:
        f = g.view

    assert g.view_copy is not None
    with native_function_manager(f):
        call_sig = DispatcherSignature.from_schema(g.view_copy.func)

        # the "view_copy" op name that the functionalization kernels need to call
        api_name = g.view_copy.func.name.unambiguous_name()
        # Sometimes the functionalization pass needs to no-op (e.g. if it was passed non-functional tensors)
        # "no-op"ing in this context is just redispatching to the original op.
        noop_api_name = f.func.name.unambiguous_name()

        dispatcher_sig = DispatcherSignature.from_schema(f.func)
        assert_view_op_properties(f.func)
        view_tensor_name = dispatcher_sig.arguments()[0].name

        return_type = dispatcher_sig.returns_type().remove_const_ref().cpp_type()

        unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(
            dispatcher_sig, is_view_op=True
        )
        view_redispatch_args = [
            e.expr
            for e in translate(unwrapped_args_ctx, call_sig.arguments(), method=False)
        ]

        forward_lambda = FunctionalizationLambda.from_func(g, is_reverse=False)
        reverse_lambda = FunctionalizationLambda.from_func(g, is_reverse=True)

        # The meta API call should use the same arguments, but convert all tensors to meta tensors first.
        meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
        meta_call_args = [
            e.expr for e in translate(meta_call_ctx, call_sig.arguments(), method=False)
        ]

        (
            symbolic_inputs_varname,
            symbolic_inputs_check,
        ) = emit_has_symbolic_inputs(call_sig)

        if "inplace_view" in f.tags:
            # See Note [Functionalization Pass - Inplace View Ops] for more details
            return f"""
    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
      if (!at::functionalization::impl::isFunctionalTensor({view_tensor_name})) {{
        // functionalization is re-entrant, but will no-op if it wasn't passed a FunctionalTensorWrapper.
        {unwrap_tensor_args_str}
        at::AutoDispatchSkipFunctionalize guard;
        return at::_ops::{noop_api_name}::call({', '.join(view_redispatch_args)});
      }}
      auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
      auto inverse_return_mode = (
          reapply_views ? at::functionalization::InverseReturnMode::ViewOrScatterInverse
            : at::functionalization::InverseReturnMode::NeverView
      );
      {symbolic_inputs_check}
      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
        {forward_lambda.decl()} {{
          if (reapply_views) {{
            return {forward_lambda.inner_call(reapply_views=True)}
          }} else {{
            return {forward_lambda.inner_call(reapply_views=False)}
          }}
        }},
        {reverse_lambda.decl()} {{
          return {reverse_lambda.inner_call()}
        }},
        /*has_symbolic_inputs=*/{symbolic_inputs_varname}
      );
      auto compute_reference_meta =
        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::XLABit) ||
        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::LazyBit);
      {return_type} reference_tensor_output;
      if (compute_reference_meta) {{
        {meta_conversion_str}
        at::AutoDispatchSkipFunctionalize func_guard;
        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
        reference_tensor_output = at::_ops::{noop_api_name}::call({', '.join(meta_call_args)});
      }}
      // This function adds the above view meta to the current tensor and replays them off the base,
      // mutating the size/stride info of the current FunctionalTensorWrapper.
      // Because of this, we need to make sure to run the reference shape function above,
      // BEFORE doing this (otherwise we'll end up runnin the reference function using the wrong sizes/strides)
      at::functionalization::impl::mutate_view_meta({view_tensor_name}, view_meta);
      // See  Note [Propagating strides in the functionalization pass]
      // XLA/LTC don't implement the logic to propagate strides correctly, so we need to rely
      // on a reference implementation here (instead of relying on the output from the forward lambda
      // having the correct stride info)
      if (compute_reference_meta) {{
        at::functionalization::impl::set_sizes_strides_offset({view_tensor_name}, reference_tensor_output);
      }}
      return {view_tensor_name};
    }}
"""

        else:
            is_multi_output_view = isinstance(f.func.returns[0].type, ListType)
            return f"""
    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
      {unwrap_tensor_args_str}
      if (!at::functionalization::impl::isFunctionalTensor({view_tensor_name})) {{
        // functionalization is re-entrant, but will no-op if it wasn't passed a FunctionalTensorWrapper.
        at::AutoDispatchSkipFunctionalize guard;
        return at::_ops::{noop_api_name}::call({', '.join(view_redispatch_args)});
      }}
      auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
      auto inverse_return_mode = (
          reapply_views ? at::functionalization::InverseReturnMode::ViewOrScatterInverse
            : at::functionalization::InverseReturnMode::NeverView
      );
      auto compute_reference_meta =
        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::XLABit) ||
        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::LazyBit);
      {return_type} reference_tensor_output;
      if (compute_reference_meta) {{
        {meta_conversion_str}
        at::AutoDispatchSkipFunctionalize func_guard;
        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
        reference_tensor_output = at::_ops::{noop_api_name}::call({', '.join(meta_call_args)});
      }}
      {return_type} tmp_output;
      {{
        at::AutoDispatchSkipFunctionalize guard;
        if (reapply_views) {{
          tmp_output = at::_ops::{noop_api_name}::call({', '.join(view_redispatch_args)});
        }} else {{
          tmp_output = at::_ops::{api_name}::call({', '.join(view_redispatch_args)});
        }}
      }}
      {symbolic_inputs_check}
      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(
        {forward_lambda.decl()} {{
          if (reapply_views) {{
            return {forward_lambda.inner_call(reapply_views=True)}
          }} else {{
            return {forward_lambda.inner_call(reapply_views=False)}
          }}
        }},
        {reverse_lambda.decl()} {{
          return {reverse_lambda.inner_call()}
        }},
        /*has_symbolic_inputs=*/{symbolic_inputs_varname},
        /*is_multi_output=*/{str(is_multi_output_view).lower()},
        /*is_as_strided=*/{str(str(f.func.name) == 'as_strided').lower()}
      );
      auto out = at::functionalization::impl::create_functional_tensor_with_view_meta(tmp_output, {view_tensor_name}, view_meta);
      // See  Note [Propagating strides in the functionalization pass]
      if (compute_reference_meta) {{
        at::functionalization::impl::set_sizes_strides_offset(out, reference_tensor_output);
      }}
      return out;
    }}
"""


def maybe_create_output(f: NativeFunction, var_name: str) -> str:
    if len(f.func.returns) == 0:
        return ""
    return_type = dispatcher.returns_type(f.func.returns).remove_const_ref().cpp_type()
    return f"{return_type} {var_name} = "


# Given a NativeFunction, and a variable name corresponding to the output of redispatching on the function,
# this returns two lists of names, consisting of:
# - the names of returns corresponding to the original (mutable) inputs of the outer function
# - the names of returns corresponding to the (immutable) outputs of the inner redispatched function
def get_mutable_redispatch_return_names(
    f: NativeFunction, inner_return_var: str
) -> Tuple[List[str], List[str]]:
    aliased_returns = []
    non_aliased_returns = []
    for i, name in enumerate(f.func.aliased_return_names()):
        if name is not None:
            aliased_returns.append(name)
        else:
            non_aliased_returns.append(
                inner_return_var
                if len(f.func.returns) == 1
                else f"std::get<{i}>({inner_return_var})"
            )
    return aliased_returns, non_aliased_returns


# When functionalization "no-op's" and redispatches on a mutable operator, we need to take care so that:
#  - For fresh outputs, we return the result of the redispatch (without wrapping outputs)
#  - For outputs that were aliased to inputs, we return the inputs directly (since some of them might have been wrapped)
def return_from_mutable_noop_redispatch(
    f: NativeFunction, inner_return_var: str
) -> str:
    aliased, non_aliased = get_mutable_redispatch_return_names(f, inner_return_var)
    # Just get all of the return names, and immediately return them
    return return_str(f.func.returns, aliased + non_aliased)


def wrap_propagate_mutations_and_return(
    f: NativeFunction, functional_op: NativeFunction, inner_return_var: str
) -> str:
    mutable_arg_names = f.func.arguments.mutable_arg_names()
    (
        aliased_outer_rets,
        non_aliased_outer_rets,
    ) = get_mutable_redispatch_return_names(f, inner_return_var)
    _, non_aliased_inner_rets = get_mutable_redispatch_return_names(
        functional_op, inner_return_var
    )
    # The outer function may have a mix of aliased and non-aliased outputs,
    # But the inner functional op that we're transforming to should only have non-aliased outputs
    assert len(mutable_arg_names) + len(non_aliased_outer_rets) == len(
        non_aliased_inner_rets
    )

    # First, take all of the newly created outputs from the inner call and wrap them into functional tensors
    updates = []
    non_aliased_wrapped_ret_names = []
    for i, inner_ret in enumerate(
        non_aliased_inner_rets[: len(non_aliased_outer_rets)]
    ):
        ret_name = f"output_{i}"
        updates.append(
            f"""\
  auto output_{i} = at::functionalization::impl::to_functional_tensor({inner_ret});"""
        )
        non_aliased_wrapped_ret_names.append(ret_name)

    # Next, take all of the mutated outputs from the inner call corresponding to mutated inputs,
    # and propagate the mutations
    for outer_arg, inner_ret in zip(
        mutable_arg_names, non_aliased_inner_rets[len(non_aliased_outer_rets) :]
    ):
        updates.append(
            f"""\
  at::functionalization::impl::propagate_xla_data({outer_arg}, {inner_ret});
  at::functionalization::impl::replace_({outer_arg}, {inner_ret});
  at::functionalization::impl::commit_update({outer_arg});
  at::functionalization::impl::sync({outer_arg});"""
        )

    # Finally, we return:
    # - Any mutable arguments that also returns
    # - Any immutable returns that were created wrapping the output from the inner call
    returns_str = return_str(
        f.func.returns, aliased_outer_rets + non_aliased_wrapped_ret_names
    )
    updates_str = "\n".join(updates)
    return f"""\
{updates_str}
    {returns_str}"""


# Generates the Functionalization kernel for:
# - mutation ops (inplace and out= ops)
@with_native_function_and
def emit_inplace_functionalization_body(
    f: NativeFunction, g: NativeFunctionsGroup
) -> str:
    # mutation case
    assert modifies_arguments(f)

    dispatcher_sig = DispatcherSignature.from_schema(f.func)

    unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(
        dispatcher_sig, is_view_op=False
    )

    mutated_names = [
        a.name
        for a in f.func.arguments.flat_all
        if a.type.is_tensor_like() and a.annotation is not None
    ]
    non_mutated_names = [
        a.name
        for a in f.func.arguments.flat_all
        if a.type.is_tensor_like() and a.annotation is None
    ]
    non_mutated_tensor_names = [
        a.name
        for a in f.func.arguments.flat_all
        if a.type == BaseType(BaseTy.Tensor) and a.annotation is None
    ]
    # all mutable inputs must be functional tensors in order to participate in functionalization
    check_all_mutated_args_are_functional = " && ".join(
        ["true"]
        + [
            f"at::functionalization::impl::isFunctionalTensor({a})"
            for a in mutated_names
        ]
    )
    check_any_non_mutated_args_are_functional = " || ".join(
        ["false"]
        + [
            f"at::functionalization::impl::isFunctionalTensor({a})"
            for a in non_mutated_names
        ]
    )

    check_any_non_mutated_tensors_are_xla = " || ".join(
        ["false"]
        + [
            f"{a}.device().type() == c10::DeviceType::XLA"
            for a in non_mutated_tensor_names
        ]
    )
    # These are used in the cases where we don't functionalize and redispatch to the inplace op
    # case 1: we hit an inplace op that doesn't have an out-of-place equivalent
    # case 2: we hit an inplace ops but our inputs are not functional tensors (in which case our kernel just no-ops)
    inplace_exprs = [
        e.expr
        for e in translate(unwrapped_args_ctx, dispatcher_sig.arguments(), method=False)
    ]

    # call the out-of-place variant of the op
    return_type = (
        dispatcher.returns_type(g.functional.func.returns).remove_const_ref().cpp_type()
    )
    functional_sig = DispatcherSignature.from_schema(g.functional.func)
    functional_exprs = [
        e.expr
        for e in translate(unwrapped_args_ctx, functional_sig.arguments(), method=False)
    ]

    if f.func.is_out_fn():
        mutable_input_post_processing = "\n".join(
            [
                f"""
      at::functionalization::impl::replace_(
        {a.name}, {'std::get<' + str(i) + '>(tmp_output)' if len(f.func.returns) > 1 else 'tmp_output'});
      at::functionalization::impl::commit_update({a.name});"""
                for (i, a) in enumerate(f.func.arguments.out)
                if a.annotation and a.annotation.is_write and a.type.is_tensor_like()
            ]
        )
    else:
        mutable_input_post_processing = "\n".join(
            [
                f"""
      at::functionalization::impl::replace_({a.name}, tmp_output);
      at::functionalization::impl::commit_update({a.name});"""
                for a in f.func.arguments.flat_all
                if a.annotation and a.annotation.is_write and a.type.is_tensor_like()
            ]
        )

    meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
    # We don't want to run the inplace meta func for ops like .set_(), because:
    # (1) they're unnecessary: inplace meta checks are only useful for ops like add_(),
    #     where broadcasting will work for the out-of-place case but should fail on the inplace call
    # (2) They'll also fail without adding extra infra: we'd need to convert the input storage argument
    #     into a meta storage
    any_storage_args = any(
        a.type == BaseType(BaseTy.Storage) for a in f.func.arguments.flat_all
    )

    return f"""
    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
      if ({str(not any_storage_args and f.func.kind() == SchemaKind.inplace).lower()}) {{
        // Before converting the mutable op to its functional variant, run meta tensors through the original op.
        // This will help us catch shape errors that apply to inplace ops that wouldn't apply to their functional variants.
        // (We can only do this for inplace ops today though, because they technically all support meta tensors).
        {meta_conversion_str}
        at::AutoDispatchSkipFunctionalize func_guard;
        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
        at::_ops::{f.func.name.unambiguous_name()}::call({', '.join(a.name for a in meta_call_ctx)});
      }}
      {unwrap_tensor_args_str}
      if (!({check_all_mutated_args_are_functional})) {{
        // We want to disable this check if there are any XLA tensors.
        // cpu_tensor.copy_(xla_tensor) is valid code.
        if (!({check_any_non_mutated_tensors_are_xla}) && ({check_any_non_mutated_args_are_functional})) {{
         // case 1: trying to mutate a non functional tensor with a functional tensor is an error
         TORCH_INTERNAL_ASSERT(false,
           "mutating a non-functional tensor with a functional tensor is not allowed.",
           " Please ensure that all of your inputs are wrapped inside of a functionalize() call.");
        }} else {{
         // case 2: arguments are not functional tensors, so we no-op and redispatch.
         at::AutoDispatchSkipFunctionalize guard;
         {maybe_create_output(f, 'tmp_output')}at::_ops::{f.func.name.unambiguous_name()}::call({', '.join(inplace_exprs)});
         {return_from_mutable_noop_redispatch(f, 'tmp_output')}
        }}
      }} else {{
        {return_type} tmp_output;
        {{
          at::AutoDispatchSkipFunctionalize guard;
          tmp_output = at::_ops::{g.functional.func.name.unambiguous_name()}::call({', '.join(functional_exprs)});
        }}
        {wrap_propagate_mutations_and_return(f, g.functional, 'tmp_output')}
      }}
    }}"""


# The below functions generate RegisterFunctionalization.cpp
# These files provide the kernels that run the functionalization pass, which can be opted into
# per backend (e.g. XLA or Vulkan), or as a composable transform (functionalize() in functorch).


# See Note [Functionalization Pass: View Inverses].
def gen_functionalization_view_inverse_declaration(
    selector: SelectiveBuilder, g: NativeFunctionsViewGroup
) -> Optional[str]:
    # For every (non-composite) view op, we need a corresponding "inverse view" function.
    # This generates the declarations so we get a good compiler error when someone adds a new view.
    @with_native_function
    def emit_decl_helper(g: NativeFunctionsViewGroup) -> Optional[str]:
        if g.view.has_composite_implicit_autograd_kernel:
            return None
        view_inverse_sig = ViewInverseSignature(g)
        return view_inverse_sig.decl()

    return emit_decl_helper(g)


def gen_functionalization_registration(
    selector: SelectiveBuilder,
    g: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup],
    composite_implicit_autograd_index: BackendIndex,
) -> List[str]:
    @with_native_function
    def emit_registration_helper(f: NativeFunction) -> str:
        assert not f.has_composite_implicit_autograd_kernel
        registration_str = f"TORCH_FN(functionalization::{wrapper_name(f.func)})"
        return f'm.impl("{f.func.name}", {registration_str});'

    # Don't generate kernels in mobile build
    if not selector.include_all_operators:
        return []

    if isinstance(g, NativeFunctionsViewGroup):
        # functionalization needs to register kernels for view + view_inplace ops
        # See Note [Functionalization <> torch.Tensor constructor]
        if str(g.view.func.name) == "lift_fresh":
            return []
        view_str = []
        if not g.view.has_composite_implicit_autograd_kernel:
            view_str.append(emit_registration_helper(g.view))
        if (
            g.view_inplace is not None
            and not g.view_inplace.has_composite_implicit_autograd_kernel
        ):
            assert g.view_inplace.is_view_op
            view_str.append(emit_registration_helper(g.view_inplace))
        return view_str

    elif isinstance(g, NativeFunctionsGroup):
        # Gets a hand-written functionalization kernel
        if g.inplace is not None and str(g.inplace.func.name) == "set_.source_Tensor":
            fns = []
        else:
            fns = list(g.functions())
    else:
        if str(g.func.name) in MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION:
            return []
        fns = [g]

    registrations = []
    for f in fns:
        if f.has_composite_implicit_autograd_kernel:
            continue
        if str(f.func.name) == "lift":
            # See Note [Functionalization <> torch.Tensor constructor]
            return []
        if str(f.func.name) == "resize_":
            # See Note [resize_ in Functionalization]
            return []
        if str(f.func.name.name) != "set_":
            assert not f.is_view_op
        # functionalization needs to generate and register kernels for inplace ops.
        # We *also* need to directly register CompositeImplicitAUtograd kernels
        # so that they decompose properly before functioanlization.
        if modifies_arguments(f):
            registrations.append(emit_registration_helper(f))
    return registrations


def gen_functionalization_definition(
    selector: SelectiveBuilder,
    # Note: Ideally this code should never have to look at NativeFunction
    # (and instead only need to operate on grouped NativeFunctions).
    # The only reason currently is because we need to emit direct dispatch registrations
    # For CompositeImplicitAutograd operators, which are potentially ungrouped.
    g: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup],
) -> List[str]:
    # Don't generate kernels in mobile build
    if not selector.include_all_operators:
        return []

    if isinstance(g, NativeFunctionsViewGroup):
        # Case 1: emit view -> view_copy kernels for the functionalization pass
        view_defs = []
        if not g.composite:
            # invariant: NativeFunctionsViewGroup's always have a view_copy operator
            # if the view is not composite (implicit autograd)
            assert g.view_copy is not None, dataclass_repr(g, indent=1)
            view_defs.append(emit_view_functionalization_body(g, view_inplace=False))
            if g.view_inplace is not None:
                view_defs.append(emit_view_functionalization_body(g, view_inplace=True))
        return view_defs
    elif isinstance(g, NativeFunction):
        # Invariant: all mutable operators that we need to handle in functionalization
        # should have been properly grouped up.
        # TODO: The below ops all have "problematic" schemas that prevent them from
        # getting functionalized. Instead of bending over backwards to get things to work,
        # I think we should either:
        # (1) fix their schemas (BC-breaking)
        # (2) hand-write their functionalization kernels
        if (
            str(g.func.name) not in MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION
            and str(g.func.name.name) not in MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION
        ):
            assert g.has_composite_implicit_autograd_kernel or not modifies_arguments(g)
        return []
    else:
        # Case 2: emit inplace -> out-of-place kernels for the functionalization pass
        mutation_defs = []
        mutation_defs.append(emit_inplace_functionalization_body(g.out, g))
        if g.inplace is not None:
            mutation_defs.append(emit_inplace_functionalization_body(g.inplace, g))
        if g.mutable is not None:
            mutation_defs.append(emit_inplace_functionalization_body(g.mutable, g))
        return mutation_defs
    return []
