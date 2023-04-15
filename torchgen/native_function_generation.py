from collections import defaultdict

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import Binding, DispatcherSignature, Expr
from torchgen.context import with_native_function
from torchgen.model import (
    Annotation,
    Argument,
    BackendIndex,
    BackendMetadata,
    BaseOperatorName,
    BaseTy,
    BaseType,
    DEFAULT_KERNEL_NAMESPACE,
    DeviceCheckType,
    DispatchKey,
    FunctionSchema,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    Return,
    SchemaKind,
    Variant,
)
from torchgen.utils import concatMap

# See Note: [Out ops with functional variants that don't get grouped properly]
OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY = [
    # This has a functional variant, but it's currently marked private.
    # This function should be marked private as well (*_backward ops aren't exposed to python anyway).
    "adaptive_avg_pool3d_backward.grad_input",
    # There's a functional variant, _slow_conv2d_backward.output_mask, that isn't grouped properly.
    # Maybe we can kill this operator in favor of convolution_backward?
    "_slow_conv2d_backward.grad_input",
]


# See Note: [Mutable ops that cannot get an out variant]
MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT = [
    # should be out=?
    "_cummax_helper",
    # should be out=?
    "_cummin_helper",
]

# All of these operators don't have any tensor like returns
FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT = [
    "_assert_async",  # no return
    "_dimI",  # returns an int
    "_dimV",  # returns an int
    "_has_same_storage_numel",  # returns a boolean
    "_linalg_check_errors",  # no return
    "_local_scalar_dense",  # returns a Scalar
    "_nested_tensor_from_mask_left_aligned",  # returns a boolean
    "_nnz",  # returns an int
    "_use_cudnn_ctc_loss",  # returns a boolean
    "_use_cudnn_ctc_loss.Tensor",  # returns a boolean
    "_validate_compressed_sparse_indices",  # no return
    "allclose",  # returns a boolean
    "dense_dim",  # returns an int
    "equal",  # returns a boolean
    "is_coalesced",  # returns an boolean
    "is_pinned",  # returns a boolean
    "is_same_size",  # returns a boolean
    "is_set_to",  # returns a boolean
    "q_per_channel_axis",  # returns an int
    "q_scale",  # returns a float
    "q_zero_point",  # returns an int
    "qscheme",  # returns a QScheme
    "record_stream",  # no return
    "sparse_dim",  # returns an int
    "_nested_tensor_storage_offsets",  # returns a vector of ints
    "_chunk_grad_outputs_efficient_attention",  # returns a bool
    "_fused_sdp_choice",  # returns an int
]

INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY = [
    # polygamma and polygamma.out both exist, but have a
    # pre-self arg (while polygamma_ does not)
    # We should either fix this schema so it can be grouped properly,
    # or allow the codegen to generate new functional/out= NativeFunctions for this op
    # (which would require changing its overload name to prevent overload ambiguity).
    "polygamma_"
]


# Groups "similar" NativeFunctions together
# example add.Tensor, add_.Tensor, add.out
# "similar" NativeFunctions are all expected to have an identical `signature()`,
# But have differing SchemaKinds.
def pre_group_native_functions(
    native_functions: Sequence[NativeFunction],
) -> Dict[FunctionSchema, Dict[SchemaKind, NativeFunction]]:
    pre_grouped_native_functions: Dict[
        FunctionSchema, Dict[SchemaKind, NativeFunction]
    ] = defaultdict(dict)
    for f in native_functions:
        d = pre_grouped_native_functions[f.func.signature()]
        assert f.func.kind() not in d
        d[f.func.kind()] = f
    return pre_grouped_native_functions


# Returns the out variant overload name given a base function overload name
def get_expected_out_variant_overload_name(overload_name: Optional[str]) -> str:
    return "out" if not overload_name else f"{overload_name}_out"


# Helper function: given an inplace FunctionSchema, generate its corresponding out= variant
# Example before:
#   _add_relu_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
# Example after:
#   _add_relu.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out)
def self_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    # Generating an out= schema from an inplace schema.
    assert func.kind() == SchemaKind.inplace
    assert func.arguments.self_arg is not None
    # The new out= schema has:
    # - a new out argument with the same type as "func" (but with a mutable annotation)
    # - The returns (if any) now alias the out= argument instead of "func"
    # - an "out" overload name
    return FunctionSchema(
        name=func.name.remove_inplace().with_overload(
            get_expected_out_variant_overload_name(func.name.overload_name)
        ),
        arguments=func.arguments.remove_self_annotation().with_out_args(
            [
                Argument(
                    name="out",
                    type=func.arguments.self_arg.argument.type,
                    default=None,
                    annotation=func.arguments.self_arg.argument.annotation,
                )
            ]
        ),
        returns=func.returns,
    )


# Helper function: given a functional FunctionSchema, generate its corresponding out= variant
# Example before:
#   _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#       bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
# Example after:
#   _to_copy._out(Tensor self, *, bool non_blocking=False, MemoryFormat? memory_format=None,
#       Tensor(a!) out) -> Tensor(a!)
def functional_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    # Generating an out= schema from a functional schema.
    assert func.kind() == SchemaKind.functional

    new_returns, new_out_args = generate_out_args_from_schema(func)
    # The new out= schema has:
    # - one or more new out argument(s) with the same type as returns (but with a mutable annotation)
    # - The returns now alias the out= arguments
    # - an "_out" overload name
    return FunctionSchema(
        name=func.name.with_overload(
            get_expected_out_variant_overload_name(func.name.overload_name)
        ),
        arguments=func.arguments.signature().with_out_args(
            new_out_args,
        ),
        returns=tuple(new_returns),
    )


# Helper function: given a function schema, generate corresponding out arguments, also the updated return annotations.
def generate_out_args_from_schema(
    func: FunctionSchema,
) -> Tuple[List[Return], List[Argument]]:
    # More of a sanity check - our existing restrictions on schemas should enforce that
    # mutable schema kinds never return their mutable arguments.
    assert not any(
        r.annotation is not None and r.annotation.is_write for r in func.returns
    )

    tensorlike_rets = [r for r in func.returns if r.type.is_tensor_like()]
    assert len(tensorlike_rets) > 0

    used_annotations = concatMap(
        lambda a: [] if a.annotation is None else a.annotation.alias_set,
        func.arguments.flat_all,
    )
    valid_annotations = [
        x for x in "abcdefghijklmnopqrstuvwxyz" if x not in used_annotations
    ]

    all_rets_are_tensors = all(r.type == BaseType(BaseTy.Tensor) for r in func.returns)

    new_out_args: List[Argument] = []
    # The end result of new_returns is that:
    # - If every return is a plain tensor, then the new returns == the old returns, but with the out= alias annotations added.
    # - Otherwise, none of the out arguments show up in the returns (and we're only left with non-tensor-like returns, if any).
    new_returns: List[Return] = []
    for i, r in enumerate(func.returns):
        if r.type.is_tensor_like():
            new_out = Argument(
                name="out" if len(func.returns) == 1 else f"out{i}",
                type=r.type,
                default=None,
                annotation=Annotation.parse(f"{valid_annotations[i]}!"),
            )
            new_out_args.append(new_out)
            if all_rets_are_tensors:
                # The convention for out= schemas is that they only return their out arguments
                # if the return is a plain Tensor (or if it's a tuple of plain Tensors)
                new_ret = Return(
                    name=None, type=new_out.type, annotation=new_out.annotation
                )
                new_returns.append(new_ret)
        else:
            new_returns.append(r)
    return new_returns, new_out_args


# Helper function: given a mutable FunctionSchema, generate its corresponding out= variant
# Example before:
#   _fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)  # noqa: B950
# Example after:
#   _fused_moving_avg_obs_fq_helper._out(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False, *, Tensor(e!) out0, Tensor(f!) out1) -> (Tensor(e!), Tensor(f!))  # noqa: B950
def mutable_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    # Generating an out= schema from a mutable schema.
    assert func.kind() == SchemaKind.mutable
    # The new out= schema has:
    # - Any non-aliased tensor-like returns are converted to mutable, aliased out= arguments
    #   (if the argument is a tensor then we also return it for method chaining,
    #   otherwise we return nothing)
    # - an "out" overload name
    #
    # Note that:
    # (1) This also means that we can *only* generate an out= variant from a mutable schema
    #     if the mutable schema has at least one tensor-like non-aliasing return.
    # (2) The generated out= variant still has mutable positional arguments,
    #     but if necessary we could probably add another out= variant that also
    #     functionalizes the mutable arguments (a functional_out variant)

    new_returns, new_out_args = generate_out_args_from_schema(func)

    return FunctionSchema(
        name=func.name.remove_inplace().with_overload(
            get_expected_out_variant_overload_name(func.name.overload_name)
        ),
        arguments=func.arguments.with_out_args(new_out_args),
        returns=tuple(new_returns),
    )


# This function, given function of one SchemaKind, as well as a target SchemaKind,
# generates a new NativeFunction with the same properties, but using the target SchemaKind.
# We only actually generate functions for either functional or out= SchemaKinds.
# This function returns a tuple, with:
# - The generated NativeFunction
# - a dictionary of `BackendIndex` objects, describing which dispatch keys
#   we will generate kernels for, for the new NativeFunction.
#   Details are in the function, but we only generate composite kernels (in some cases) today.
def generate_function(
    f: NativeFunction, k: SchemaKind
) -> Tuple[NativeFunction, Dict[DispatchKey, Dict["OperatorName", "BackendMetadata"]]]:
    from torchgen.api import cpp

    if k == SchemaKind.functional:
        assert f.func.kind() != SchemaKind.functional
        # The new "functional" NativeFunction has:
        # - any mutable arguments have been converted into (immutable) returns.
        #   (if a mutable argument was not also a return, it gets converted to one)
        # - "_functional" appended to the base name, ONLY IF this op has a mutable variant.
        #   See Note [Overload Ambiguity With Functional Variants]
        # The default grouping logic in signature() actually already does this,
        # so we can piggy-back off it (but we still want return names)
        func = f.func.signature(keep_return_names=True).with_name(
            OperatorName(
                name=BaseOperatorName(
                    base=f.func.name.name.base,
                    inplace=False,
                    dunder_method=f.func.name.name.dunder_method,
                    # See Note [Overload Ambiguity With Functional Variants]
                    functional_overload=f.func.kind() == SchemaKind.mutable,
                ),
                overload_name=f.func.name.overload_name,
            )
        )
    elif k == SchemaKind.out:
        # We generate out= ops mostly just so that we can pair up NativeFunctions into groups easily,
        # but at least today, there is no good reason to actually use them.
        # we'll generate a dispatcher entry for them, but won't actually register any kernels for them.
        if f.func.kind() == SchemaKind.inplace:
            func = self_to_out_signature(f.func)
        elif f.func.kind() == SchemaKind.mutable:
            func = mutable_to_out_signature(f.func)
        elif f.func.kind() == SchemaKind.functional:
            func = functional_to_out_signature(f.func)
        else:
            raise AssertionError(
                "We only bother generating out= functions from either inplace or mutable or functional variants"
            )
    else:
        raise AssertionError(
            "We currently only generate either functional or out= NativeFunctions"
        )

    # Generated kernel naming convention for out: <op_name>_<overload_name>. The reason for this is to
    # disambiguate operator with the same name but different overload name, e.g., `randn.names_out` and
    # `randn.generator_with_names_out`.
    kernel_name = (
        func.name.unambiguous_name()
        if func.kind() == SchemaKind.out
        else cpp.name(func)
    )
    if f.func.has_symint():
        kernel_name += "_symint"
    backend_metadata = {
        DispatchKey.CompositeExplicitAutograd: {
            func.name: BackendMetadata(
                kernel=kernel_name,
                structured=False,
                cpp_namespace=DEFAULT_KERNEL_NAMESPACE,
            )
        }
    }
    tags = {"generated"} | set(f.tags & {"nondeterministic_seeded", "view_copy"})

    return (
        NativeFunction(
            func=func,
            use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors,
            # These generated fn's aren't meant to be user friendly- don't generate methods.
            variants={Variant.function},
            structured=False,
            structured_delegate=None,
            structured_inherits=None,
            precomputed=None,
            autogen=[],
            ufunc_inner_loop={},
            manual_kernel_registration=False,
            manual_cpp_binding=False,
            python_module=None,
            category_override=None,
            device_guard=False,
            device_check=DeviceCheckType.NoCheck,
            loc=f.loc,
            cpp_no_default_args=set(),
            is_abstract=f.is_abstract,
            has_composite_implicit_autograd_kernel=False,
            has_composite_implicit_autograd_nested_tensor_kernel=False,
            has_composite_explicit_autograd_kernel=True,
            has_composite_explicit_autograd_non_functional_kernel=False,
            # Every generated NativeFunction gets a "generated" tag, so it's easy to tell
            # which NativeFunction objects did not come directly from native_functions.yaml.
            tags=tags,
            namespace=f.namespace,
        ),
        backend_metadata,
    )


# This function is responsible for adding generated NativeFunctions which don't appear
# explicitly in the codegen.
# You can inspect the full list of NativeFunctions yourself with the torchgen package, by running
# torchgen.parse_native_yaml("aten/src/ATen/native/native_functions.yaml", "aten/src/ATen/native/tags.yaml")
# (Maybe we should make a friendly API for this)
#
# Note: this function *mutates* its two inputs,
# adding the new NativeFunctions / BackendMetadata to them
def add_generated_native_functions(
    rs: List[NativeFunction],
    indices: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]],
) -> None:
    # The main code for gnerating new NativeFunctions
    # First we group of NaitveFunctions by schema kind,
    # then we detect which ones are missing and generate them.
    pre_grouped_native_functions = pre_group_native_functions(rs)
    for k, d in pre_grouped_native_functions.items():
        has_functional = SchemaKind.functional in d
        has_inplace = SchemaKind.inplace in d
        has_mutable = SchemaKind.mutable in d
        has_out = SchemaKind.out in d

        # We automatically generate a few native functions that don't exist in the yaml, for a few reasons:
        # (1) If an operator has an inplace/out= variant but no functional variant, we can generate
        #     a simple functional variant that the functionalization pass can consume.
        # (2) If an operator has an inplace or functional but no out= variant, we generate an out=
        #     variant, mostly so we can easily pair up functions into NativeFunctionsGroup,
        #     while maintaining the constraint that the out= variant is "required".
        if has_mutable or has_inplace or has_out or has_functional:
            # Don't bother generating functions trio's for native functions that bypass the dispatcher.
            are_manual = all(f.manual_cpp_binding for f in d.values())
            # Don't bother generating functional + out= variants for view operators
            has_view_ops = any(f.is_view_op for f in d.values())
            # Don't generate the other variants for CompositeImplicitAutograd operators.
            # We could probably do this, but the main benefit of generating the function triplets
            # is for transforms that need them, and transforms don't need to act directly
            # on CompositeImplicitAutograd operators (since we let them decompose).
            are_composite_implicit = all(
                f.has_composite_implicit_autograd_kernel for f in d.values()
            )
            if are_manual or has_view_ops or are_composite_implicit:
                continue
            if has_out and len(d.values()) == 1:
                # Note: [Out ops with functional variants that don't get grouped properly]
                # In theory we could validly have an out= operator in native_functions.yaml
                # that has no other variants.
                # But today, all of the operators where that's the case actually do have
                # functional variants, that we are just unable to pair up properly.
                # I think banning this all together is probably safer
                # (you can always add a functional variant yourself if you want to add a new out= operator).
                #
                # We should probably fix the existing cases; this check is to prevent us from adding more over time.
                if (
                    str(d[SchemaKind.out].func.name)
                    not in OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY
                ):
                    raise AssertionError(
                        f"Found an out= operator that we could not find any other variants of: {str(d[SchemaKind.out].func)}"
                    )
                continue

            # Some inplace ops that have problematic schemas (that we should fix), which prevent us
            # from generating out= and functional variants
            if (
                has_inplace
                and str(d[SchemaKind.inplace].func.name)
                in INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY
            ):
                continue

            base_fn = (
                d[SchemaKind.inplace]
                if has_inplace
                else d[SchemaKind.mutable]
                if has_mutable
                else d[SchemaKind.out]
                if has_out
                else d[SchemaKind.functional]
            )

            # Note: [Mutable ops that cannot get an out variant]
            # We can only generate an out= variant if either:
            # - the original function has tensor-like returns (since we can convert them to out kwargs)
            # - or it's inplace (since we can convert `self` to an out kwarg)
            # There are only two functions that don't fit this criteria today though,
            # and they both look like they should be fixed to be out= variants,
            # so if feels safer to ban this schema all-together
            base_fn_valid = base_fn.func.kind() == SchemaKind.inplace or any(
                r.type.is_tensor_like() for r in base_fn.func.returns
            )
            # Note: [Loosen the assertion that all functional should have out variant]
            # By design all functional operators should have our variants. The needs_out check
            # is loosening this requirement, changing it to only generate out variant if there's
            # an `autogen` block in the native function, in the long run it should be removed.
            # FIXME: Remove this after figuring out CI job failures related to min, max, mean
            needs_out = any("out" in str(op_name) for op_name in base_fn.autogen)
            gets_out_variant = not has_out and base_fn_valid and needs_out
            if not has_out and not base_fn_valid:
                if (
                    str(base_fn.func.name)
                    not in MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT
                    and str(base_fn.func.name)
                    not in FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT
                ):
                    raise AssertionError(
                        f"""Found an operator that we could not generate an out= variant for: {str(base_fn.func)}.
This type of operators don't have tensor-like return, making it difficult to generate a proper out= variant. If
out= variant is not needed, please add the function name into FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT list."""
                    )

            # Generate an out= variant
            if gets_out_variant:
                fn, metadata = generate_function(base_fn, SchemaKind.out)
                d[SchemaKind.out] = fn
                BackendIndex.grow_index(indices, metadata)
                rs.append(fn)

            # Generate a functional variant, but only do it if the operator got an out= variant
            # (Functional variants are only useful if we can group up the variants,
            # which we can only do if they have an out= variant)
            if not has_functional and (has_out or gets_out_variant):
                fn, metadata = generate_function(base_fn, SchemaKind.functional)
                d[SchemaKind.functional] = fn
                BackendIndex.grow_index(indices, metadata)
                rs.append(fn)


def return_str(rets: Tuple[Return, ...], names: List[str]) -> str:
    assert len(rets) == len(names)
    if len(rets) == 0:
        return ""
    elif len(rets) == 1:
        return f"return {names[0]};"
    else:
        return f"return {dispatcher.returns_type(rets).cpp_type()}({', '.join(names)});"


# Given a function, and the name of a variable correponding to the output of that function,
# gather up all of the individual returns that are not aliased
def gather_nonaliased_inner_rets(func: FunctionSchema, out_var: str) -> List[str]:
    aliased_rets = func.aliased_return_names()
    non_aliased_names = []
    is_out_var_a_tuple = len(func.returns) > 1
    for i, r in enumerate(aliased_rets):
        if r is None:
            non_aliased_names.append(
                f"std::get<{i}>({out_var})" if is_out_var_a_tuple else out_var
            )
    return non_aliased_names


# Generates functional kernels in terms of their inplace.mutable counterparts.
# We only do this for "generated" NativeFunctions
@with_native_function
def gen_composite_functional_kernel(g: NativeFunctionsGroup) -> Optional[str]:
    # We should only be generating these for code-generated NativeFunctions
    if "generated" not in g.functional.tags:
        return None
    # And we always write the kernel for a generated op in terms of a non-generated op.
    if g.inplace is not None and "generated" not in g.inplace.tags:
        target_f = g.inplace
    elif g.mutable is not None and "generated" not in g.mutable.tags:
        target_f = g.mutable
    else:
        # We should be guaranteed to have a valid inplace/mutable variant to call into.
        # See Note: [Mutable Ops Not Using Functionalization]
        raise AssertionError(str(g.functional.func))

    sig = DispatcherSignature(g.functional.func)
    target_sig = DispatcherSignature(target_f.func)

    context: List[Union[Binding, Expr]] = []
    clone_mutable_inputs = []
    cloned_return_names = []
    # We can't just directly pass all of the arguments from the functional op into the mutating op.
    # We need to check for which inputs to the mutating operator are mutable,
    # and clone those inputs first.
    for a_curr, a_tgt in zip(
        dispatcher.jit_arguments(g.functional.func),
        dispatcher.jit_arguments(target_f.func),
    ):
        if a_tgt.annotation is not None and a_tgt.annotation.is_write:
            clone_mutable_inputs.append(
                f"auto {a_curr.name}_clone = clone_arg({a_curr.name});"
            )
            context.append(
                Expr(
                    expr=f"{a_curr.name}_clone",
                    type=dispatcher.argument_type(a_curr, binds=a_curr.name),
                )
            )
            # Invariant: mutable arguments on the inner mutable op are always returns on the functional op.
            cloned_return_names.append(f"{a_curr.name}_clone")
        else:
            context.append(dispatcher.argument(a_curr))
    exprs = ", ".join([e.expr for e in translate(context, target_sig.arguments())])

    out_name = "output"
    maybe_assign = f"auto {out_name} = " if len(target_f.func.returns) > 0 else ""
    inner_return_names = gather_nonaliased_inner_rets(target_f.func, out_name)
    ret_str = return_str(
        g.functional.func.returns, inner_return_names + cloned_return_names
    )

    clone_mutable_inputs_str = "\n".join(clone_mutable_inputs)
    return f"""
{sig.defn(name=sig.name() + ("_symint" if g.out.func.has_symint() else ""))} {{
  {clone_mutable_inputs_str}
  {maybe_assign}at::_ops::{target_f.func.name.unambiguous_name()}::call({exprs});
  {ret_str}
}}
"""


# Generates out= kernels in terms of their functional counterparts.
# We only do this for "generated" NativeFunctions
@with_native_function
def gen_composite_out_kernel(g: NativeFunctionsGroup) -> Optional[str]:
    # We should only be generating these for code-generated NativeFunctions
    if "generated" not in g.out.tags:
        return None
    # And we always write the kernel for the out= op in terms of the functional.
    # Note that the functional op might have also been generated, but we don't have to
    # worry about cycles, because the generated functional kernels are always implemented
    # in terms of non-generated kernels (see gen_composite_functional_kernel).

    sig = DispatcherSignature(g.out.func)
    target_sig = DispatcherSignature(g.functional.func)

    exprs = ", ".join(
        [e.expr for e in translate(sig.arguments(), target_sig.arguments())]
    )

    copy_outs = []
    out_name = "tmp_output"
    for i, out_arg in enumerate(g.out.func.arguments.out):
        functional_return_name = (
            out_name
            if len(g.functional.func.returns) == 1
            else f"std::get<{i}>({out_name})"
        )
        copy_outs.append(
            f"""\
  resize_out_helper({out_arg.name}, {functional_return_name});
  copy_arg({out_arg.name}, {functional_return_name});"""
        )

    rets = []
    # For each return arg in the calling (out=) operator,
    # If it corresponds to an aliased input, return the input.
    # Otherwise, return the corresponding output from calling the functional operator.
    for i, ret_name in enumerate(g.out.func.aliased_return_names()):
        if ret_name is not None:
            rets.append(ret_name)
        else:
            functional_return_name = (
                out_name
                if len(g.functional.func.returns) == 1
                else f"std::get<{i}>({out_name})"
            )
            rets.append(functional_return_name)

    copy_outs_str = "\n".join(copy_outs)

    # Kernel name needs to follow the naming convention defined in `generate_function()`
    return f"""
{sig.defn(name=g.out.func.name.unambiguous_name() + ("_symint" if g.out.func.has_symint() else ""))} {{
  auto {out_name} = at::_ops::{g.functional.func.name.unambiguous_name()}::call({exprs});
  {copy_outs_str}
  {return_str(g.out.func.returns, rets)}
}}
"""
