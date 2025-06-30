# Generates ADInplaceOrViewType.h/cpp
#
# NOTE: If any changes are being made to the ADInplaceOrView codegen please also check
# if updates are needed in torch/csrc/autograd/autograd_not_implemented_fallback.cpp
# The fallback is expected to mimic this codegen, so we should keep the two in sync.

from __future__ import annotations

from torchgen.api import cpp
from torchgen.api.autograd import (
    dispatch_strategy,
    gen_differentiable_outputs,
    NativeFunctionWithDifferentiabilityInfo,
)
from torchgen.api.types import (
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    CType,
    DispatcherSignature,
    intArrayRefT,
    longT,
    OptionalCType,
    symIntArrayRefT,
    SymIntT,
    tensorT,
)
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import (
    NativeFunction,
    SchemaKind,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)
from torchgen.utils import FileManager

from .context import with_native_function_with_differentiability_info
from .gen_trace_type import (
    get_return_value,
    MANUAL_AUTOGRAD,
    tie_return_values,
    type_wrapper_name,
)


# See NOTE [ Autograd View Variables ] in variable.h for details.
# If you update list VIEW_FUNCTIONS or RETURNS_VIEWS_OF_INPUT,
# you **MUST** also update the public list of view ops accordingly in
# docs/source/tensor_view.rst. Note not all ATen functions are exposed to public,
# e.g alias & sparse_coo_tensor_with_dims_and_tensors.
#
# A map: function name => name of the argument that all outputs are view of

VIEW_FUNCTIONS_WITH_METADATA_CHANGE = [
    "view_as_complex",
    "view_as_real",
    "_conj",
    "_neg_view",
    "_nested_get_values",
    "_nested_view_from_buffer",
    "_nested_view_from_jagged",
]

VIEW_FUNCTIONS = {
    "numpy_T": "self",
    "alias": "self",
    "as_strided": "self",
    "diagonal": "self",
    "expand": "self",
    "permute": "self",
    "select": "self",
    "slice": "self",
    "slice_inverse": "self",
    "split": "self",
    "split_with_sizes": "self",
    "squeeze": "self",
    "t": "self",
    "transpose": "self",
    "unfold": "self",
    "unsqueeze": "self",
    "flatten": "self",
    "view": "self",
    "unbind": "self",
    "_indices": "self",
    "_values": "self",
    "indices": "self",
    "values": "self",
    "crow_indices": "self",
    "col_indices": "self",
    "ccol_indices": "self",
    "row_indices": "self",
    # sparse_coo ctor output should really be views of both indices and values,
    # but we only supports making as view of a single variable, and indices is
    # discrete anyways.
    # FIXME: clone indices on construction.
    "sparse_coo_tensor_with_dims_and_tensors": "values",
    "_reshape_alias": "self",
    "_test_autograd_multiple_dispatch_view": "self",
}

for key in VIEW_FUNCTIONS_WITH_METADATA_CHANGE:
    VIEW_FUNCTIONS[key] = "self"

# note: some VIEW_FUNCTIONS are just compositions of the view functions above
# this list contains both the root view functions and any that are purely composed
# of viewing functions, and is used by the JIT to determine when an operator
# may return a view of its inputs; however they may sometimes return a copy.
# (e.g. `contiguous`)
RETURNS_VIEWS_OF_INPUT = set(VIEW_FUNCTIONS.keys()).union(
    {
        "chunk",
        "detach",
        "contiguous",
        "reshape",
        "reshape_as",
        "expand_as",
        "view_as",
        "real",
        "imag",
        "narrow",
        "movedim",
        "tensor_split",
        "swapdims",
        "swapaxes",
        "mT",
        "mH",
        "adjoint",
        "matrix_H",
    }
)

# These are the functions we consider views for the purposes of validating
# StorageImpl and TensorImpl in gen_variable_type.
# `_unsafe_view` is not included in VIEW_FUNCTIONS above because it is not a
# view for the purposes of ADInplaceOrView kernel, we do not want to call as_view
# See NOTE [Unsafe View] for more info.
ALL_VIEW_FUNCTIONS = {
    **VIEW_FUNCTIONS,
    "_unsafe_view": "self",
}

ARRAYREF_TO_VEC = CodeTemplate(
    """\
auto ${vec} = ${arg}.vec();
"""
)

OPTIONAL_TO_VAL = CodeTemplate(
    """\
auto ${val} = ${arg}.value_or(${default});
"""
)

CALL_DISPATCH = CodeTemplate(
    """\
at::_ops::${unambiguous_name}::call(${unpacked_args})"""
)

REVERSE_VIEW_DISPATCH = CodeTemplate(
    """\
${reverse_name}(${unpacked_args})"""
)

MULTI_OUTPUT_VIEW_ITERATION = CodeTemplate(
    """\
for (auto ${view_idx} : c10::irange(${var}.size())) {
  ${body}
}
"""
)

SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE = CodeTemplate(
    """\
std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
if (${is_view_with_metadata_change} ||
    !self.unsafeGetTensorImpl()->support_as_strided() ||
    self.unsafeGetTensorImpl()->is_python_dispatch() ||
    c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
  ${replay_view_func}
  ${reverse_replay_view_func}
}
"""
)

REPLAY_VIEW_FUNC = CodeTemplate(
    """\
func = std::make_unique<${view_func_name}>(${view_func_args});
"""
)

REVERSE_REPLAY_VIEW_LAMBDA_FUNC = CodeTemplate(
    """\
rev_func = [=](const at::Tensor& ${input_view}) {
  return ${reverse_replay_view_call};
};
"""
)

METHOD_DEFINITION = CodeTemplate(
    """\
${return_type} ${type_wrapper_name}(${formals}) {
  ${type_definition_body}
}
"""
)

WRAPPER_REGISTRATION = CodeTemplate(
    """\
m.impl("${unqual_operator_name_with_overload}",
       TORCH_FN(${class_type}::${type_wrapper_name})
);
"""
)

AUTOGRAD_NOT_IMPLEMENTED_REGISTRATION = CodeTemplate(
    """\
m.impl("${unqual_operator_name_with_overload}", torch::autograd::autogradNotImplementedFallback());
"""
)

INPLACE_REDISPATCH = CodeTemplate(
    """\
{
  at::AutoDispatchBelowADInplaceOrView guard;
  at::_ops::${unambiguous_name}::redispatch(${unpacked_args});
}
"""
)

ASSIGN_RETURN_VALUE = CodeTemplate(
    """\
${return_values} = ${rhs_value};
"""
)

VIEW_REDISPATCH = CodeTemplate(
    """\
${assign_return_values} ([&]() {
  at::AutoDispatchBelowADInplaceOrView guard;
  return at::_ops::${unambiguous_name}::redispatch(${unpacked_args});
})();
"""
)

TMP_VAR = "_tmp"


# FIXME: Ideally these functions should be methods on Type class, but we have a
#        comment in codegen/model.py there saying these concepts are not well defined.
#        Thus we put a version that commonly used by autograd codegen here.
def is_tensor_type(t: Type) -> bool:
    # TODO: Should handle optional here?
    return t.is_tensor_like() and t.is_list_like() is None


def is_tensor_list_type(t: Type) -> bool:
    # TODO: Should handle optional here?
    return t.is_tensor_like() and t.is_list_like() is not None


UNPACK_TENSOR = CodeTemplate(
    """\
auto${ref} ${arg_name}_ = unpack${suffix}(${arg_name}, "${arg_name}", ${arg_pos});"""
)


def unpacked_name(arg_name: str) -> str:
    return arg_name + "_"


# e.g. select.int -> select_copy_int_inverse()
def inverse_view_name(f: NativeFunction) -> str:
    copy_variant = f"{f.root_name}_copy"
    overload = f"{f.func.name.overload_name}"
    if overload != "":
        overload = "_" + overload
    return f"{copy_variant}{overload}_inverse"


def extract_bindings(f: NativeFunction) -> list[Binding]:
    return [
        r
        for a in f.func.schema_order_arguments()
        for r in cpp.argument(
            a,
            method=False,
            symint=True,
            cpp_no_default_args=set(),
            faithful=False,
            has_tensor_options=False,
        )
    ]


@with_native_function
def unpack_args(f: NativeFunction) -> tuple[list[str], list[Binding]]:
    body: list[str] = []
    unpacked_bindings: list[Binding] = []

    for i, binding in enumerate(extract_bindings(f)):
        assert not isinstance(binding.argument, SelfArgument)
        if isinstance(binding.argument, TensorOptionsArguments):
            raise RuntimeError("VariableKernel shouldn't take TensorOptions")

        is_nullable = binding.argument.type.is_nullable()
        if not binding.argument.type.is_tensor_like() or is_nullable:
            unpacked_bindings.append(binding)
            continue

        is_tensor_list = is_tensor_list_type(binding.argument.type)
        ref = (not is_nullable) and not is_tensor_list
        suffix = "_opt" if is_nullable and not is_tensor_list else ""
        body.append(
            UNPACK_TENSOR.substitute(
                arg_name=binding.name,
                arg_pos=i,
                suffix=suffix,
                ref="&" if ref else "",
            )
        )
        unpacked_bindings.append(
            Binding(
                name=unpacked_name(binding.name),
                nctype=binding.nctype,
                argument=binding.argument,
                default=binding.default,
            )
        )

    return body, unpacked_bindings


def get_base_name(f: NativeFunction) -> str:
    return f.func.name.name.base  # TODO: should be str(f.func.name.name)?


def get_view_info(f: NativeFunction) -> str | None:
    base_name = get_base_name(f)
    view_info = VIEW_FUNCTIONS.get(base_name, None)
    if view_info is None and base_name in RETURNS_VIEWS_OF_INPUT:
        view_info = "self"
    return view_info


def emit_view_func(
    f: NativeFunction, bindings: list[Binding], view_idx: str | None = None
) -> str:
    """Generate an additional lambda function to recover views in backward when as_strided is not supported.
    See Note [View + Inplace update for base tensor] and [View + Inplace update for view tensor] for more details.
    """
    # TODO: Clean this logic up if we get rid of reverse view funcs or reify them.
    input_base = "input_base"
    replay_view_func = ""
    updated_args: list[str] = []
    known_view_arg_simple_types: list[CType] = [
        BaseCType(longT),
        OptionalCType(BaseCType(longT)),
        BaseCType(SymIntT),
        OptionalCType(BaseCType(SymIntT)),
        BaseCType(boolT),
        BaseCType(intArrayRefT),
        BaseCType(symIntArrayRefT),
        ConstRefCType(BaseCType(tensorT)),
        ConstRefCType(OptionalCType(BaseCType(tensorT))),
    ]
    for binding in bindings:
        arg, arg_type = binding.name, binding.nctype.type
        if arg == "self":
            updated_args.append(input_base)
            continue
        if arg_type not in known_view_arg_simple_types:
            known_types_str = ", ".join([str(t) for t in known_view_arg_simple_types])
            raise TypeError(
                f"You are adding an {arg_type} {arg} argument to op {cpp.name(f.func)} in addition to known types: "
                f"{known_types_str}. Please update the list or materialize it so that it can be closed "
                "over by value, also add a test in pytorch/xla/test/test_operations.py where this code "
                "is exercised."
            )
        if arg_type == BaseCType(intArrayRefT) or arg_type == BaseCType(
            symIntArrayRefT
        ):
            # It's not safe to close over IntArrayRef by value, since this is a
            # reference type, so materialize a vector to close over by value
            arg_vec = arg + "_vec"
            replay_view_func += ARRAYREF_TO_VEC.substitute(arg=arg, vec=arg_vec)
            updated_args.append(arg_vec)
        elif arg_type == OptionalCType(BaseCType(longT)):
            # Materialize int64_t? to int64_t
            arg_value = arg + "_val"
            replay_view_func += OPTIONAL_TO_VAL.substitute(
                arg=arg, val=arg_value, default="0"
            )
            updated_args.append(arg_value)
        elif arg_type == ConstRefCType(BaseCType(tensorT)) or arg_type == ConstRefCType(
            OptionalCType(BaseCType(tensorT))
        ):
            # NB: Closing over a tensor. If a user modifies this tensor, this will be silently
            # incorrect. The proper thing to do is to store the version counter and copy on write.
            updated_args.append(arg)
        else:
            updated_args.append(arg)

    from .gen_view_funcs import view_func_name

    view_func_args = [b.name for b in bindings if b.name != "self"]
    if view_idx is not None:
        view_func_args.append(f"{view_idx}")
    replay_view_func += REPLAY_VIEW_FUNC.substitute(
        view_func_name=view_func_name(f, include_namespace=True),
        view_func_args=view_func_args,
    )

    input_view = "input_view"
    reverse_unpacked_args = [
        "self",
        f"{input_view}",
        # inverse_return_mode=
        "at::functionalization::InverseReturnMode::AlwaysView",
        *(() if view_idx is None else (f"{view_idx}",)),
        # skip input_base arg
        *updated_args[1:],
    ]

    from torchgen.api.functionalization import reverse_name

    reverse_replay_view_call = REVERSE_VIEW_DISPATCH.substitute(
        reverse_name=reverse_name(f, include_namespace=True),
        unpacked_args=reverse_unpacked_args,
    )
    reverse_replay_view_func = REVERSE_REPLAY_VIEW_LAMBDA_FUNC.substitute(
        input_view=input_view, reverse_replay_view_call=reverse_replay_view_call
    )

    is_view_with_metadata_change = (
        "true" if cpp.name(f.func) in VIEW_FUNCTIONS_WITH_METADATA_CHANGE else "false"
    )

    return SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE.substitute(
        is_view_with_metadata_change=is_view_with_metadata_change,
        replay_view_func=replay_view_func,
        reverse_replay_view_func=reverse_replay_view_func,
    )


def emit_view_body(
    fn: NativeFunctionWithDifferentiabilityInfo, var: str
) -> tuple[str, str]:
    # See NOTE [ Autograd View Variables ] in variable.h for details.
    f = fn.func
    base_name = get_base_name(f)
    view_info = get_view_info(f)
    call = ""
    differentiable_outputs = gen_differentiable_outputs(fn)
    differentiable_output_vars = {r.name for r in differentiable_outputs}
    if not isinstance(view_info, str):
        raise TypeError(
            f"The view info should be a string for {base_name}, but it is: {view_info}"
        )
    if len(differentiable_output_vars) == 0:
        # no output is differentiable (.indices() for SparseTensors for example)
        rhs_value = (
            f"as_view({view_info}, {var}, "
            f"/* is_bw_differentiable */ false, /* is_fw_differentiable */ false)"
        )
    elif len(differentiable_output_vars) == 1:
        # Single differentiable output (Tensor or Tensor[])
        return_info = differentiable_outputs[0]
        # We only support simple Tensor or a TensorList for functions that return views
        if not is_tensor_type(return_info.type) and not is_tensor_list_type(
            return_info.type
        ):
            raise RuntimeError(
                f"{base_name} that return differentiable views can only return Tensor or Tensor[]"
            )

        # See Note [ View + Inplace detection]
        def get_creation_meta_in_mode(original: str) -> str:
            creation_meta_with_grad_mode = f"(at::GradMode::is_enabled() ? {original} : CreationMeta::NO_GRAD_MODE)"
            return f"InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : {creation_meta_with_grad_mode}"

        # Only allow rebasing of the history if we return a single Tensor
        # If we are in a no grad block, raise a warning
        # See NOTE [ View + Inplace detection ] for more details about this logic
        if is_tensor_list_type(return_info.type):
            creation_meta = get_creation_meta_in_mode("CreationMeta::MULTI_OUTPUT_NODE")
            view_idx = "view_idx"
            view_func = emit_view_func(
                f, extract_bindings(f), view_idx=view_idx
            ).strip()
            as_view_call = (
                f"as_view(/* base */ {view_info}, /* output */ {var}[{view_idx}], "
                "/* is_bw_differentiable */ true, /* is_fw_differentiable */ true, "
                "/* view_func */ std::move(func), /* rev_view_func */ rev_func, "
                f"/* creation_meta */ {creation_meta});"
            )
            call += MULTI_OUTPUT_VIEW_ITERATION.substitute(
                var=var, view_idx=view_idx, body=f"{view_func}\n{as_view_call}"
            )
            rhs_value = f"std::move({var})"
        else:
            call += emit_view_func(f, extract_bindings(f), view_idx=None)
            creation_meta = get_creation_meta_in_mode("CreationMeta::DEFAULT")
            rhs_value = (
                f"as_view(/* base */ {view_info}, /* output */ {var}, /* is_bw_differentiable */ true, "
                "/* is_fw_differentiable */ true, "
                f"/* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ {creation_meta})"
            )
    else:
        # This could be supported but we don't need it at the moment, so keeping things simple.
        raise RuntimeError(
            "Function that return multiple differentiable output "
            "when at least one of them is view is not supported."
        )
    return call, rhs_value


def modifies_arguments(f: NativeFunction) -> bool:
    return f.func.kind() in [SchemaKind.inplace, SchemaKind.out]


@with_native_function_with_differentiability_info
def emit_inplace_or_view_body(fn: NativeFunctionWithDifferentiabilityInfo) -> list[str]:
    f = fn.func
    inplace_view_body: list[str] = []

    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_exprs = dispatcher_sig.exprs()

    # code-generated ADInplaceOrView kernels plumb and recompute dispatch keys directly through the kernel for performance.
    # See Note [Plumbing Keys Through The Dispatcher] for details.
    dispatch_key_set = "ks & c10::after_ADInplaceOrView_keyset"
    redispatch_args = ", ".join([dispatch_key_set] + [a.expr for a in dispatcher_exprs])

    # Note that this calls the slow, dispatching variants of manual_cpp_binding ops.
    # We could probably work harder to ensure that the fast variants are called instead, but the perf benefit would be minimal.
    if modifies_arguments(f):  # inplace op
        inplace_view_body.append(
            INPLACE_REDISPATCH.substitute(
                unambiguous_name=f.func.name.unambiguous_name(),
                unpacked_args=redispatch_args,
            )
        )
        for r in cpp.return_names(f):
            inplace_view_body.append(f"increment_version({r});")
    else:
        assert get_view_info(f) is not None
        inplace_view_body.append(
            VIEW_REDISPATCH.substitute(
                assign_return_values="auto " + TMP_VAR + " = ",
                unambiguous_name=f.func.name.unambiguous_name(),
                unpacked_args=redispatch_args,
            )
        )
        call, rhs_value = emit_view_body(fn, TMP_VAR)
        inplace_view_body.append(call)
        assert rhs_value is not None
        inplace_view_body.append(
            ASSIGN_RETURN_VALUE.substitute(
                return_values=tie_return_values(f), rhs_value=rhs_value
            )
        )
    if f.func.returns:
        inplace_view_body.append(f"return {get_return_value(f)};")
    return inplace_view_body


@with_native_function
def gen_formals(f: NativeFunction) -> str:
    return ", ".join(
        # code-generated autograd kernels plumb and recompute dispatch keys directly through the kernel for performance.
        # See Note [Plumbing Keys Through The Dispatcher] for details.
        ["c10::DispatchKeySet ks"]
        + [
            f"{cpp.argument_type(a, binds='__placeholder__', symint=True).cpp_type()} {a.name}"
            for a in f.func.schema_order_arguments()
        ]
    )


@with_native_function_with_differentiability_info
def inplace_or_view_method_definition(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> str | None:
    f = fn.func
    if get_view_info(f) is None and (
        # For functions that modify their inputs but don't return them,
        # we can't give them autograd support.
        # See https://github.com/pytorch/pytorch/issues/53796
        not modifies_arguments(f) or len(f.func.returns) == 0
    ):
        return None
    return METHOD_DEFINITION.substitute(
        return_type=cpp.returns_type(f.func.returns, symint=True).cpp_type(),
        type_wrapper_name=type_wrapper_name(f),
        formals=gen_formals(f),
        type_definition_body=emit_inplace_or_view_body(fn),
    )


@with_native_function_with_differentiability_info
def inplace_or_view_method_registration(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> str | None:
    f = fn.func
    if get_view_info(f) is None and (
        not modifies_arguments(f) or len(f.func.returns) == 0
    ):
        return None
    return WRAPPER_REGISTRATION.substitute(
        unqual_operator_name_with_overload=f.func.name,
        type_wrapper_name=type_wrapper_name(f),
        class_type="ADInplaceOrView",
    )


def use_derived(fn: NativeFunctionWithDifferentiabilityInfo) -> bool:
    f = fn.func
    name = cpp.name(f.func)
    return name not in MANUAL_AUTOGRAD and dispatch_strategy(fn) == "use_derived"


def gen_inplace_or_view_type_env(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> dict[str, list[str]]:
    definition = inplace_or_view_method_definition(fn)
    registration = inplace_or_view_method_registration(fn)

    return {
        "ops_headers": (
            [f"#include <ATen/ops/{fn.func.root_name}_ops.h>"]
            if definition is not None
            else []
        ),
        "inplace_or_view_method_definitions": [definition]
        if definition is not None
        else [],
        "inplace_or_view_wrapper_registrations": [registration]
        if registration is not None
        else [],
    }


def gen_inplace_or_view_type(
    out: str,
    native_yaml_path: str,
    tags_yaml_path: str,
    fns_with_infos: list[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
) -> None:
    # NOTE: see Note [Sharded File] at the top of the VariableType.cpp
    # template regarding sharding of the generated files.

    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_sharded(
        "ADInplaceOrViewType.cpp",
        [fn for fn in fns_with_infos if use_derived(fn)],
        key_fn=lambda fn: fn.func.root_name,
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/ADInplaceOrViewType.cpp",
        },
        env_callable=gen_inplace_or_view_type_env,
        num_shards=2,
        sharded_keys={
            "ops_headers",
            "inplace_or_view_method_definitions",
            "inplace_or_view_wrapper_registrations",
        },
    )
