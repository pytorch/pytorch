from torchgen.model import (
    Argument,
    DispatchKey,
    FunctionSchema,
    BaseType,
    BaseTy,
    Return,
    Annotation,
    NativeFunction,
    OperatorName,
    BackendIndex,
    BackendMetadata,
    DeviceCheckType,
    SchemaKind,
    Variant,
)
from torchgen.utils import (
    concatMap,
)


from typing import List, Tuple, Sequence, Dict
from collections import defaultdict

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
            "out" if not func.name.overload_name else f"{func.name.overload_name}_out"
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


# Helper function: given a mutable FunctionSchema, generate its corresponding out= variant
# Example before:
#   _fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)  # noqa: B950
# Example after:
#   _fused_moving_avg_obs_fq_helper.out(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False, *, Tensor(e!) out0, Tensor(f!) out1) -> (Tensor(e!), Tensor(f!))  # noqa: B950
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
    for (i, r) in enumerate(func.returns):
        if r.type.is_tensor_like():
            new_out = Argument(
                name=f"out{i}",
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

    return FunctionSchema(
        name=func.name.remove_inplace().with_overload(
            "out" if not func.name.overload_name else f"{func.name.overload_name}_out"
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
        gets_composite_kernel = True
        # The new "functional" NativeFunction has:
        # - any mutable arguments have been converted into (immutable) returns.
        #   (if a mutable argument was not also a return, it gets converted to one)
        # - a "functional" overload name.
        # The default grouping logic in signature() actually already does this,
        # so we can piggy-back off it (but we still want return names)
        func = f.func.signature(keep_return_names=True).with_name(
            f.func.name.remove_inplace().with_overload(
                "functional"
                if not f.func.name.overload_name
                else f"{f.func.name.overload_name}_functional"
            )
        )
    elif k == SchemaKind.out:
        # We generate out= ops mostly just so that we can pair up NativeFunctions into groups easily,
        # but at least today, there is no good reason to actually use them.
        # we'll generate a dispatcher entry for them, but won't actually register any kernels for them.
        gets_composite_kernel = False
        if f.func.kind() == SchemaKind.inplace:
            func = self_to_out_signature(f.func)
        elif f.func.kind() == SchemaKind.mutable:
            func = mutable_to_out_signature(f.func)
        else:
            raise AssertionError(
                "We only bother generating out= functions from either inplace or mutable variants"
            )
    else:
        raise AssertionError(
            "We currently only generate either functional or out= NativeFunctions"
        )

    if gets_composite_kernel:
        backend_metadata = {
            DispatchKey.CompositeExplicitAutograd: {
                func.name: BackendMetadata(cpp.name(func), structured=False)
            }
        }
    else:
        backend_metadata = {}

    return (
        NativeFunction(
            func=func,
            use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors,
            # These generated fn's aren't meant to be user friendly- don't generate methods.
            variants=set([Variant.function]),
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
            has_composite_explicit_autograd_kernel=gets_composite_kernel,
            # Every generated NativeFunction gets a "generated" tag, so it's easy to tell
            # which NativeFunction objects did not come directly from native_functions.yaml.
            tags=set(["generated"]),
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
        # (2) If an operator has an inplace and functional but no out= variant, we generate an out=
        #     variant, mostly so we can easily pair up functions into NativeFunctionsGroup,
        #     while maintaining the constraint that the out= variant is "required".
        #
        # For now, we don't bother generated NativeFunctions for existing operators
        # that only have a functional variant.
        if has_mutable or has_inplace or has_out:

            # Don't bother generating functions trio's for native functions that bypass the dispatcher.
            are_manual = all(f.manual_cpp_binding for f in d.values())
            # Don't bother generating functional + out= variants for view operators
            has_view_ops = (
                has_inplace and "inplace_view" in d[SchemaKind.inplace].tags
            ) or any(f.is_view_op for f in d.values())
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
            )

            # Note: [Mutable ops that cannot get an out variant]
            # We can only generate an out= variant if either:
            # - the original function has tensor-like returns (since we can convert them to out kwargs)
            # - or it's inplace (since we can convert `self` to an out kwarg)
            # There are only two functions that don't fit this criteria today though,
            # and they both look like they should be fixed to be out= variants,
            # so if feels safer to ban this schema all-together
            gets_out_variant = not has_out and (
                base_fn.func.kind() == SchemaKind.inplace
                or any(r.type.is_tensor_like() for r in base_fn.func.returns)
            )
            if not has_out and not gets_out_variant:
                if (
                    str(base_fn.func.name)
                    not in MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT
                ):
                    raise AssertionError(
                        f"""Found a mutable operator that we could not generate an out= variant for: {str(base_fn.func)}.
These operators are problematic, because we can't easily auto-generate functionalization code for them. If you really need
the operator have the schema mentioned, that add the name of the operator to the allow-list. Otherwise if possible,
please convert it to an inplace operator"""
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
