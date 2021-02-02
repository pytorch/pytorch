# Generates VariableType.h/cpp
#
# VariableType is a subclass of at::Type that provides the binding code
# necessary to provide a differentiable version of ATen operators. There are a
# number of different things we could mean:
#
#   - Given a non-differentiable forward implementation, we might
#     directly associate it with a backward implementation to make
#     it differentiable.  This is the common case.
#
#   - Some functions don't need a backwards implementation, because
#     backpropagation will never propagate beyond them.  There are a
#     number of different reasons why this may be the case:
#
#       - The function has no differentiable inputs
#       - The function's output is not differentiable
#       - The function has no data dependency on its input
#
#   - Some function don't need a backwards implementation because they
#     are implemented as a composition of other (differentiable) ATen
#     functions.  These are dispatched directly to the Type superclass,
#     which will in turn dispatch back to VariableType for its
#     differentiable subcomponents.
#
from dataclasses import dataclass

from .gen_autograd import VIEW_FUNCTIONS, VIEW_FUNCTIONS_WITH_METADATA_CHANGE, \
    MULTI_OUTPUT_SAFE_FUNCTIONS, RETURNS_VIEWS_OF_INPUT
from .gen_autograd_functions import uses_single_grad
from .gen_trace_type import (
    MANUAL_BACKEND, MANUAL_AUTOGRAD_AND_TRACER, MANUAL_AUTOGRAD,
    declare_returned_variables, tie_return_values, get_return_value, type_wrapper_name,
)

from tools.codegen.api.types import *
from tools.codegen.api.autograd import *
import tools.codegen.api.cpp as cpp
from tools.codegen.code_template import CodeTemplate
from tools.codegen.gen import parse_native_yaml, FileManager
from tools.codegen.context import with_native_function
from tools.codegen.utils import mapMaybe
from tools.codegen.model import *
from tools.codegen.selective_build.selector import SelectiveBuilder
from typing import Callable, List, Optional, Sequence, Tuple, Union

# We don't set or modify grad_fn on these methods. Generally, they return
# tensors that have requires_grad=False. In-place functions listed here will
# not examine or modify requires_grad or grad_fn.
DONT_REQUIRE_DERIVATIVE = {
    # These only depend on the input Tensor's shape and device, not the data
    'ones_like', 'zeros_like', 'rand_like', 'randn_like',
    # These are only implemented on integral types
    '__and__', '__iand__', '__ilshift__', '__ior__', '__irshift__', '__ixor__',
    '__lshift__', '__or__', '__rshift__', '__xor__',
    # These work on integral data types, and hence don't require derivative
    '_sobol_engine_draw', '_sobol_engine_ff', '_sobol_engine_scramble_',
    '_sobol_engine_initialize_state_',
    # This is an unsafe method that is meant to be out of reach of autograd.
    '_coalesced_',
    # Quantize functions should not record gradients
    'quantize_per_tensor', 'quantize_per_channel',
    # Functions that return integers should not have output that require gradients
    'argmax', 'argmin', 'argsort', 'searchsorted',
    'bucketize',
    # Functions that return booleans are not differentiable
    'isnan', 'isposinf', 'isneginf', 'isinf'
    # Functions return none are not differentiable
    'record_stream',
}

# The C -> R functions at the time of adding this are still being audited and tested
# but will not error out.
# C -> C, R -> C functions for which backward is correctly implemented and tested
GRADIENT_IMPLEMENTED_FOR_COMPLEX = {
    't', 'view', 'reshape', 'reshape_as', 'view_as', 'roll', 'clone',
    'repeat', 'expand', 'flip', 'fliplr', 'flipud', 'rot90', 'transpose',
    'permute', 'squeeze', 'unsqueeze', 'resize', 'resize_as', 'tril',
    'triu', 'chunk', 'zero_', 'eq_', 'ne_', 'add', '__radd__', 'sum',
    '_conj', 'sin', 'cos', 'mul', 'sinc', 'sinh', 'cosh', '__rmul__',
    'sgn', 'asin', 'acos', 'sub', 'div', 'cat', 'view_as_complex',
    'neg', 'complex', 'select', '_s_where', 'as_strided', 'slice', 'constant_pad_nd',
    'unbind', 'split', 'split_with_sizes', 'unsafe_split', 'split_with_sizes_backward',
    'dot', 'vdot', 'cholesky', 'triangular_solve', 'mm', '_unsafe_view', 'mv', 'ger',
    'bmm', 'diagonal', 'alias', 'atan', 'log', 'log10', 'log1p', 'log2', 'reciprocal',
    'tan', 'pow', 'rsqrt', 'tanh', 'tanh_backward', 'asinh', 'acosh', 'atanh', 'take', 'fill_',
    'exp', 'nonzero', 'mean', 'inverse', 'solve', 'linalg_cholesky', 'addcmul', 'addcdiv',
    'matrix_exp', 'linalg_eigh', 'cholesky_solve', 'linalg_qr', '_svd_helper', '_fft_c2c', '_fft_r2c',
    'linalg_solve', 'sqrt', 'stack', 'gather', 'index_select', 'index_add_', 'linalg_inv',
    'l1_loss_backward', 'baddbmm', 'addbmm', 'addmm', 'addmv', 'addr',
    'constant_pad_nd', 'reflection_pad1d', 'reflection_pad2d',
    'reflection_pad1d_backward', 'reflection_pad2d_backward',
    'replication_pad1d', 'replication_pad2d', 'replication_pad3d',
    'replication_pad1d_backward', 'replication_pad2d_backward', 'replication_pad3d_backward',
    'masked_scatter', 'masked_select',
    'index_fill',
}

# Some operators invalidate the grad_accumulator. Let's reset it.
RESET_GRAD_ACCUMULATOR = {
    'set', 'resize'
}

# NOTE [ Invariant: TensorImpl and Storage Pointer Equality ]
#
# When a function modifies its input tensors (via inplace or out-variants),
# it should never change the the input tensors' underlying c10::TensorImpl pointers
# or c10::Storage pointers.
#
# The following code templates implement the checks for this invariant:
SAVE_TENSOR_STORAGE = CodeTemplate("""\
c10::optional<Storage> ${tensor_name}_storage_saved =
  ${tensor_name}.has_storage() ? c10::optional<Storage>(${tensor_name}.storage()) : c10::nullopt;
""")

ENFORCE_SAME_TENSOR_STORAGE = CodeTemplate("""\
if (${tensor_name}_storage_saved.has_value())
  AT_ASSERT(${tensor_name}_storage_saved.value().is_alias_of(${tensor_name}.storage()));
""")

SAVE_TENSORLIST_STORAGE = CodeTemplate("""\
std::vector<c10::optional<Storage>> ${tensorlist_name}_storage_saved(${tensorlist_name}.size());
for (const Tensor& tensor : ${tensorlist_name})
  ${tensorlist_name}_storage_saved.push_back(
    tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
""")

ENFORCE_SAME_TENSORLIST_STORAGE = CodeTemplate("""\
for (size_t i=0; i<${tensorlist_name}.size(); i++) {
  if (${tensorlist_name}_storage_saved[i].has_value())
    AT_ASSERT(${tensorlist_name}_storage_saved[i].value().is_alias_of(${tensorlist_name}[i].storage()));
}
""")

SAVE_OPTIONALTENSORLIST_STORAGE = CodeTemplate("""\
std::vector<c10::optional<Storage>> ${tensorlist_name}_storage_saved(${tensorlist_name}.size());
for (const c10::optional<Tensor>& tensor : ${tensorlist_name})
  ${tensorlist_name}_storage_saved.push_back(
    tensor.has_value() && tensor->has_storage() ? c10::optional<Storage>(tensor->storage()) : c10::nullopt);
""")

ENFORCE_SAME_OPTIONALTENSORLIST_STORAGE = CodeTemplate("""\
for (size_t i=0; i<${tensorlist_name}.size(); i++) {
  if (${tensorlist_name}_storage_saved[i].has_value())
    AT_ASSERT(${tensorlist_name}_storage_saved[i].value().is_alias_of(
        static_cast<c10::optional<Tensor>>(${tensorlist_name}[i])->storage()));
}
""")

SAVE_TENSOR_IMPL = CodeTemplate("""\
c10::intrusive_ptr<TensorImpl> ${tensor_name}_impl_saved;
if (${tensor_name}.defined()) ${tensor_name}_impl_saved = ${tensor_name}.getIntrusivePtr();
""")

ENFORCE_SAME_TENSOR_IMPL = CodeTemplate("""\
if (${tensor_name}_impl_saved) AT_ASSERT(${tensor_name}_impl_saved == ${tensor_name}.getIntrusivePtr());
""")

SAVE_TENSORLIST_IMPL = CodeTemplate("""\
std::vector<c10::intrusive_ptr<TensorImpl>> ${tensorlist_name}_impl_saved(${tensorlist_name}.size());
for (size_t i=0; i<${tensorlist_name}.size(); i++)
  if (${tensorlist_name}[i].defined()) ${tensorlist_name}_impl_saved[i] = ${tensorlist_name}[i].getIntrusivePtr();
""")

ENFORCE_SAME_TENSORLIST_IMPL = CodeTemplate("""\
for (size_t i=0; i<${tensorlist_name}.size(); i++) {
  if (${tensorlist_name}_impl_saved[i])
    AT_ASSERT(${tensorlist_name}_impl_saved[i] == ${tensorlist_name}[i].getIntrusivePtr());
}
""")

SAVE_OPTIONALTENSORLIST_IMPL = CodeTemplate("""\
std::vector<c10::intrusive_ptr<TensorImpl>> ${tensorlist_name}_impl_saved(${tensorlist_name}.size());
for (size_t i=0; i<${tensorlist_name}.size(); i++) {
  c10::optional<Tensor> t = ${tensorlist_name}[i];
  if (t.has_value() && t->defined()) ${tensorlist_name}_impl_saved[i] = t->getIntrusivePtr();
}
""")

ENFORCE_SAME_OPTIONALTENSORLIST_IMPL = CodeTemplate("""\
for (size_t i=0; i<${tensorlist_name}.size(); i++) {
  if (${tensorlist_name}_impl_saved[i])
    AT_ASSERT(${tensorlist_name}_impl_saved[i] == static_cast<c10::optional<Tensor>>(${tensorlist_name}[i])->getIntrusivePtr());
}
""")

# The following list contains functions that we don't enforce the invariant on.
DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE = {
    # These functions are expected to change impl or storage of input tensors
    'set_', '_cudnn_rnn_flatten_weight',
}
# END CHECKS FOR [ Invariant: TensorImpl and Storage Pointer Equality ]

METHOD_DECLARATION = CodeTemplate("""\
${return_type} ${type_wrapper_name}(${formals}) ;
""")

METHOD_DEFINITION = CodeTemplate("""\
${return_type} ${type_wrapper_name}(${formals}) {
  ${type_definition_body}
}
""")

WRAPPER_REGISTRATION = CodeTemplate("""\
m.impl("${unqual_operator_name_with_overload}",
       TORCH_FN(${class_type}::${type_wrapper_name})
);
""")

UNPACK_TENSOR = CodeTemplate("""\
auto${ref} ${arg_name}_ = unpack${suffix}(${arg_name}, "${arg_name}", ${arg_pos});""")

DECLARE_GRAD_FN = CodeTemplate("""\
std::shared_ptr<${op}> grad_fn;
""")

SETUP_ANY_REQUIRES_GRAD = CodeTemplate("""\
auto _any_requires_grad = compute_requires_grad( ${args_with_derivatives} );
(void)_any_requires_grad;
""")

SETUP_DERIVATIVE = CodeTemplate("""\
if (_any_requires_grad) {
  ${setup}
}
""")

SETUP_NONE_REQUIRES_GRAD = CodeTemplate("""\
if (compute_requires_grad( ${args_to_check} )) {
  throw_error_out_requires_grad("${base_name}");
}
""")

ASSIGN_GRAD_FN = CodeTemplate("""\
grad_fn = std::shared_ptr<${op}>(new ${op}(${op_ctor}), deleteNode);
grad_fn->set_next_edges(collect_next_edges( ${args_with_derivatives} ));
""")

CALL_DISPATCH_VIA_NAMESPACE = CodeTemplate("""\
at::${api_name}(${unpacked_args})""")

CALL_DISPATCH_VIA_METHOD = CodeTemplate("""\
${var}.${api_name}(${unpacked_method_args})""")

# If the non-variable operation has return values, we use the `tmp` variable to hold the
# values temporarily and pass the values to the return variables outside of the
# `at::AutoNonVariableTypeMode` guard block.
DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES = CodeTemplate("""\
auto tmp = ([&]() {
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  return ${base_type_call};
})();
""")

ASSIGN_RETURN_VALUE = CodeTemplate("""\
${return_values} = ${rhs_value};
""")

ARRAYREF_TO_VEC = CodeTemplate("""\
auto ${vec} = ${arg}.vec();
""")

OPTIONAL_TO_VAL = CodeTemplate("""\
auto ${val} = ${arg}.value_or(${default});
""")

SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE = CodeTemplate("""\
std::function<at::Tensor(const at::Tensor&)> func=nullptr;
if (${is_view_with_metadata_change} || !self.unsafeGetTensorImpl()->support_as_strided()) {
  ${replay_view_func}
}
""")

REPLAY_VIEW_LAMBDA_FUNC = CodeTemplate("""\
func = [=](const at::Tensor& ${input_base}) {
  return ${replay_view_call};
};
""")

DISPATCH_TO_NON_VAR_TYPE_WITHOUT_RETURN_VALUES = CodeTemplate("""\
{
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  ${base_type_call};
}
""")

SET_HISTORY = CodeTemplate("""\
if (grad_fn) {
    ${fn}_history(${differentiable_outputs}, grad_fn);
}
""")

CONDITIONAL = CodeTemplate("""\
if (${cond}) {
  ${statements}
}
""")

RUN_ONLY_IN_DEBUG_MODE = CodeTemplate("""\
#ifndef NDEBUG
${statements}
#endif
""")

@dataclass(frozen=True)
class NativeFunctionWithDifferentiabilityInfo:
    func: NativeFunction
    info: Optional[DifferentiabilityInfo]

def gen_variable_type(
    out: str,
    native_yaml_path: str,
    differentiability_infos: Sequence[DifferentiabilityInfo],
    template_path: str,
    operator_selector: SelectiveBuilder,
) -> None:

    """VariableType.h and VariableType.cpp body

    This is the at::Type subclass for differentiable tensors. The
    implementation of each function dispatches to the base tensor type to
    compute the output. The grad_fn is attached to differentiable functions.
    """
    fns = list(sorted(filter(
        operator_selector.is_native_function_selected_for_training,
        parse_native_yaml(native_yaml_path)), key=lambda f: cpp.name(f.func)))
    fns_with_infos = match_differentiability_info(fns, differentiability_infos)

    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    gen_variable_type_shard(fm, fns_with_infos, 'VariableType.h', 'VariableType.h')

    # NOTE: see Note [Sharded File] at the top of the VariableType.cpp
    # template regarding sharding of the generated files.
    num_shards = 5
    shards: List[List[NativeFunctionWithDifferentiabilityInfo]] = [[] for _ in range(num_shards)]

    # functions are assigned arbitrarily but stably to a file based on hash
    for fn in fns_with_infos:
        x = sum(ord(c) for c in cpp.name(fn.func.func)) % num_shards
        shards[x].append(fn)

    for i, shard in enumerate(shards):
        gen_variable_type_shard(fm, shard, 'VariableType.cpp', f'VariableType_{i}.cpp')

    gen_variable_type_shard(fm, fns_with_infos, 'VariableType.cpp', 'VariableTypeEverything.cpp')

@with_native_function
def gen_formals(f: NativeFunction) -> str:
    return ', '.join(
        f'{cpp.argument_type(a, binds="__placeholder__").cpp_type()} {a.name}'
        for a in f.func.schema_order_arguments()
    )

@with_native_function
def gen_wrapper_registration(f: NativeFunction) -> str:
    return WRAPPER_REGISTRATION.substitute(
        unqual_operator_name_with_overload=f.func.name,
        type_wrapper_name=type_wrapper_name(f),
        class_type='VariableType',
    )

def gen_variable_type_shard(
    fm: FileManager,
    fns_with_infos: List[NativeFunctionWithDifferentiabilityInfo],
    template_name: str,
    output_name: str,
) -> None:
    type_declarations: List[str] = []
    type_definitions: List[str] = []
    wrapper_registrations: List[str] = []

    for fn in fns_with_infos:
        f = fn.func
        name = cpp.name(f.func)
        formals = gen_formals(f)

        type_declarations.append(METHOD_DECLARATION.substitute(
            return_type=cpp.returns_type(f.func.returns),
            type_wrapper_name=type_wrapper_name(f),
            formals=formals,
        ))

        if name not in MANUAL_AUTOGRAD and dispatch_strategy(fn) == 'use_derived':
            type_definitions.append(METHOD_DEFINITION.substitute(
                return_type=cpp.returns_type(f.func.returns),
                type_wrapper_name=type_wrapper_name(f),
                type_definition_body=emit_body(fn),
                formals=formals,
            ))
            wrapper_registrations.append(gen_wrapper_registration(f))

        # See Note [Manual Backend kernels]
        assert (name in MANUAL_BACKEND) == f.manual_kernel_registration
        # If you want to register a kernel to Autograd, you must make the op abstract.
        # In other words, this op must have dispatch section in native_functions.yaml.
        if name in MANUAL_AUTOGRAD_AND_TRACER or (fn.info and fn.info.has_derivatives):
            msg = (f'There\'s a formula for {name}(or its functional variant) in derivatives.yaml. '
                   f'It\'s required to add a dispatch section for it with explicit supported backends e.g CPU/CUDA '
                   f'or DefaultBackend in native_functions.yaml. Please see '
                   f'https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native#choosing-the-right-dispatch-keyword '
                   f'for instructions to choose the right dispatch keyword.')
            assert f.is_abstract, msg

    fm.write_with_template(output_name, template_name, lambda: {
        'generated_comment': '@' + f'generated from {fm.template_dir}/{template_name}',
        'type_derived_method_declarations': type_declarations,
        'type_derived_method_definitions': type_definitions,
        'wrapper_registrations': wrapper_registrations,
    })

def emit_body(fn: NativeFunctionWithDifferentiabilityInfo) -> List[str]:
    assert dispatch_strategy(fn) == 'use_derived'
    f = fn.func
    info = fn.info

    name = cpp.name(f.func)
    inplace = f.func.kind() == SchemaKind.inplace
    is_out_fn = f.func.kind() == SchemaKind.out
    returns_void = len(f.func.returns) == 0
    base_name = f.func.name.name.base  # TODO: should be str(f.func.name.name)?
    view_info = VIEW_FUNCTIONS.get(base_name, None)
    if view_info is None and base_name in RETURNS_VIEWS_OF_INPUT:
        view_info = "self"

    def is_differentiable(name: str, type: Type) -> bool:
        return type.is_tensor_like() and (info is None or name not in info.non_differentiable_arg_names)

    def gen_differentiable_input(
        arg: Union[Argument, SelfArgument, TensorOptionsArguments]
    ) -> Optional[DifferentiableInput]:
        if isinstance(arg, TensorOptionsArguments):
            return None
        a: Argument = arg.argument if isinstance(arg, SelfArgument) else arg

        # TODO: `cpp_type` is only to keep it byte-for-byte compatible with the old codegen, should remove.
        # NB: This is not a clone of cpp.argument() - TensorOptionsArguments / faithful / binds are
        # not handled properly as they are irrelevant for this codegen.
        cpp_type = cpp.argument_type(a, binds=a.name).cpp_type()

        if not is_differentiable(a.name, a.type):
            return None
        return DifferentiableInput(
            name=a.name,
            type=a.type,
            cpp_type=cpp_type,
        )

    @with_native_function
    def gen_differentiable_inputs(f: NativeFunction) -> List[DifferentiableInput]:
        return list(mapMaybe(gen_differentiable_input, f.func.arguments.non_out))

    def find_args_with_derivatives(differentiable_inputs: List[DifferentiableInput]) -> List[DifferentiableInput]:
        """Find arguments that have derivative definitions"""
        if info is None or not info.has_derivatives:
            return differentiable_inputs
        names = set(name for d in info.derivatives for name in d.var_names)
        differentiable = [arg for arg in differentiable_inputs if arg.name in names]
        if len(differentiable) != len(names):
            missing = names - set(arg.name for arg in differentiable)
            raise RuntimeError(f'Missing arguments for derivatives: {missing} in {info.name}')
        return differentiable

    def gen_differentiable_outputs(f: NativeFunction) -> List[DifferentiableOutput]:
        outputs: List[DifferentiableOutput] = [
            DifferentiableOutput(name=name, type=ret.type, cpp_type=cpp.return_type(ret))
            for name, ret in zip(cpp.return_names(f), f.func.returns)]

        output_differentiability = info.output_differentiability if info else None
        if output_differentiability is not None:
            differentiable_outputs: List[DifferentiableOutput] = []
            if False in output_differentiability and f.func.kind() == SchemaKind.inplace:
                raise RuntimeError("output_differentiability=False for inplace operation (version_counter won't get updated)")
            for differentiable, output in zip(output_differentiability, outputs):
                if differentiable:
                    differentiable_outputs.append(output)
            return differentiable_outputs

        candidate_differentiable_outputs = list(filter(lambda r: is_differentiable(r.name, r.type), outputs))

        if uses_single_grad(info):
            return candidate_differentiable_outputs[:1]
        else:
            return candidate_differentiable_outputs

    differentiable_inputs = gen_differentiable_inputs(f)
    args_with_derivatives = find_args_with_derivatives(differentiable_inputs)
    differentiable_outputs = gen_differentiable_outputs(f)

    requires_derivative = (
        base_name not in DONT_REQUIRE_DERIVATIVE and name not in DONT_REQUIRE_DERIVATIVE and
        len(differentiable_inputs) > 0 and len(differentiable_outputs) > 0)

    if info is not None and info.has_derivatives and not requires_derivative:
        raise RuntimeError(f'ERROR: derivative ignored for {name} -- specified an autograd function without derivative')

    def emit_save_inputs() -> List[str]:
        setup: List[str] = []
        if info is None or not info.has_derivatives:
            return setup

        has_tensorlist_arg = any(is_tensor_list_type(arg.type) for arg in args_with_derivatives)

        # We don't want to save tensors if we know that they will never be used
        # when computing the derivative, so we add guards to those statements
        def guard_for(arg: SavedAttribute) -> Optional[str]:
            assert info is not None

            # It's hard to determine the edge offset if we have TensorLists
            if has_tensorlist_arg:
                return None

            # Empirical evaluation of the cases where we insert those guards in
            # backward show that they are somewhat useless. E.g. there's no need
            # to guard on some values captured from forward, because they had to
            # require_grad if the backward function even gets executed. I don't
            # have any good ideas for detecting those cases, so I simply disabled the
            # checks.
            if 'backward' in info.name:
                return None

            # If there's a single derivative we could compute, we already have
            # a requires_grad check that is sufficient
            if len(args_with_derivatives) <= 1:
                return None

            # We really only care about trimming down the amount of tensors we save
            if arg.type != 'Tensor':
                return None

            # We want to emit simple guards, so we only allow that if checking one
            # input is enough to determine whether we need that value
            used_in = [d for d in info.derivatives if arg in d.saved_inputs]
            assert len(used_in) > 0
            if len(used_in) != 1:
                return None
            derivative = used_in[0]
            if len(derivative.var_names) != 1:
                return None
            derivative_var_name = derivative.var_names[0]

            # Figure out the offset of the edge that uses this variable
            for edge_off, a in enumerate(args_with_derivatives):
                if a.name == derivative_var_name:
                    break
            else:
                raise AssertionError()

            return f'grad_fn->should_compute_output({edge_off})'

        setup.extend(save_variables(info.all_saved_inputs, False, guard_for))
        for arg in args_with_derivatives:
            if is_tensor_list_type(arg.type):
                setup.append(f'grad_fn->{arg.name}_size_ = {arg.name}.size();')

        return setup

    def setup_derivative(differentiable_inputs: List[DifferentiableInput]) -> List[str]:
        body: List[str] = []
        if is_out_fn:
            # For out functions, ensure that no input or output requires grad
            body.append(DECLARE_GRAD_FN.substitute(op='Node'))
            body.append(SETUP_NONE_REQUIRES_GRAD.substitute(
                base_name=base_name,
                args_to_check=[arg.name for arg in differentiable_inputs]))
            body.append(SETUP_NONE_REQUIRES_GRAD.substitute(
                base_name=base_name,
                args_to_check=[arg.name for arg in differentiable_outputs]))
            return body

        op = info.op if info is not None and info.has_derivatives else 'NotImplemented'
        setup = []
        setup.extend(ASSIGN_GRAD_FN.substitute(
            op=op,
            op_ctor='' if info is not None and info.has_derivatives else f'"{cpp.name(f.func)}"',
            args_with_derivatives=[arg.name for arg in args_with_derivatives],
        ).split('\n'))
        setup.extend(emit_save_inputs())

        body.extend(emit_check_no_requires_grad(differentiable_inputs, args_with_derivatives))
        body.append(DECLARE_GRAD_FN.substitute(op=op))
        body.append(SETUP_DERIVATIVE.substitute(setup=setup))
        return body

    def emit_check_if_in_complex_autograd_allowlist() -> List[str]:
        body: List[str] = []
        if base_name in GRADIENT_IMPLEMENTED_FOR_COMPLEX:
            return body
        for arg in differentiable_outputs:
            name = arg.name
            # TODO: should be `arg.type.is_tensor_like()`?
            if arg.cpp_type in ['Tensor', 'TensorList', 'const c10::List<c10::optional<Tensor>> &']:
                body.append(f'throw_error_for_complex_autograd({name}, "{base_name}");')
        return body

    def emit_check_no_requires_grad(
        tensor_args: List[DifferentiableInput],
        args_with_derivatives: List[DifferentiableInput],
    ) -> List[str]:
        """Checks that arguments without derivatives don't require grad"""
        body: List[str] = []
        for arg in tensor_args:
            if arg in args_with_derivatives:
                continue
            name = arg.name
            if info and name in info.non_differentiable_arg_names:
                continue
            if name == 'output':
                # Double-backwards definitions sometimes take in 'input' and
                # 'output', but only define the derivative for input.
                continue
            body.append(f'check_no_requires_grad({name}, "{name}");')
        return body

    def save_variables(
        saved_variables: Sequence[SavedAttribute],
        is_output: bool,
        guard_for: Callable[[SavedAttribute], Optional[str]] = lambda name: None,
    ) -> Sequence[str]:
        # assign the saved variables to the generated grad_fn
        stmts: List[str] = []
        for arg in saved_variables:
            name = arg.name
            expr = arg.expr
            if arg.type == 'Tensor' or arg.type == 'c10::optional<Tensor>' or \
                    arg.type == 'c10::optional<Tensor>&' or (is_output and arg.type == 'Scalar'):
                name += '_'
                var = arg.name
                if var == 'self' and inplace:
                    var = 'self.clone()'
                    assert not is_output
                if inplace and is_output:
                    var = 'self'
                    is_inplace_view = f'{var}.is_view()'
                    expr = f'SavedVariable({var}, {str(is_output).lower()}, {is_inplace_view})'
                else:
                    expr = f'SavedVariable({var}, {str(is_output).lower()})'
            elif arg.type in ['TensorList', 'c10::List<c10::optional<Tensor>>']:
                name += '_'
                expr = f'make_saved_variable_list({arg.name})'
            elif arg.type == 'IntArrayRef':
                expr = expr + ".vec()"
            guard = guard_for(arg)
            if guard is None:
                stmts.append(f'grad_fn->{name} = {expr};')
            else:
                stmts.append(f'if ({guard}) {{')
                stmts.append(f'  grad_fn->{name} = {expr};')
                stmts.append('}')
        return stmts

    def emit_dispatch_call(f: NativeFunction, input_base: str, unpacked_args: Sequence[str]) -> str:
        """ Dispatch call via function in a namespace or method on Tensor."""
        if Variant.function in f.variants:
            call = CALL_DISPATCH_VIA_NAMESPACE.substitute(
                api_name=cpp.name(
                    f.func,
                    faithful_name_for_out_overloads=True,
                ),
                unpacked_args=unpacked_args)
        else:
            call = CALL_DISPATCH_VIA_METHOD.substitute(
                api_name=cpp.name(f.func),
                var=input_base,
                unpacked_method_args=unpacked_args[1:])
        return call

    def emit_view_lambda(unpacked_bindings: List[Binding]) -> str:
        """ Generate an additional lambda function to recover views in backward when as_strided is not supported.
        See Note [View + Inplace update for base tensor] and [View + Inplace update for view tensor] for more details."""
        input_base = 'input_base'
        replay_view_func = ''
        updated_unpacked_args: List[str] = []
        known_view_arg_simple_types: List[str] = ['int64_t', 'c10::optional<int64_t>', 'bool', 'IntArrayRef']
        for unpacked_binding in unpacked_bindings:
            arg, arg_type = unpacked_binding.name, unpacked_binding.type
            if arg == 'self_':
                updated_unpacked_args.append(input_base)
                continue
            if arg_type not in known_view_arg_simple_types:
                known_types_str = ', '.join(known_view_arg_simple_types)
                raise TypeError(f'You are adding an {arg_type} {arg} argument to op {cpp.name(f.func)} in addition to known types: '
                                f'{known_types_str}. Please update the list or materialize it so that it can be closed '
                                'over by value, also add a test in pytorch/xla/test/test_operations.py where this code '
                                'is exercised.')

            if arg_type == 'IntArrayRef':
                # It's not safe to close over IntArrayRef by value, since this is a
                # reference type, so materialize a vector to close over by value
                arg_vec = arg + '_vec'
                replay_view_func += ARRAYREF_TO_VEC.substitute(arg=arg, vec=arg_vec)
                updated_unpacked_args.append(arg_vec)
            elif arg_type == 'c10::optional<int64_t>':
                # Materialize int64_t? to int64_t
                arg_value = arg + '_val'
                replay_view_func += OPTIONAL_TO_VAL.substitute(arg=arg, val=arg_value, default='0')
                updated_unpacked_args.append(arg_value)
            else:
                updated_unpacked_args.append(arg)

        replay_view_call = emit_dispatch_call(f, input_base, updated_unpacked_args)
        replay_view_func += REPLAY_VIEW_LAMBDA_FUNC.substitute(
            input_base=input_base,
            replay_view_call=replay_view_call)

        is_view_with_metadata_change = 'true' if name in VIEW_FUNCTIONS_WITH_METADATA_CHANGE else 'false'

        return SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE.substitute(
            is_view_with_metadata_change=is_view_with_metadata_change,
            replay_view_func=replay_view_func)

    def wrap_output(f: NativeFunction, unpacked_bindings: List[Binding], var: str) -> str:
        call = ''
        rhs_value: Optional[str] = None
        if not any(r.type.is_tensor_like() for r in f.func.returns):
            rhs_value = var
        elif view_info is not None:
            # See NOTE [ Autograd View Variables ] in variable.h for details.
            differentiable_output_vars = {r.name for r in differentiable_outputs}

            if not isinstance(view_info, str):
                raise TypeError(f'The view info should be a string for {base_name}, but it is: {view_info}')

            if len(differentiable_output_vars) == 0:
                # no output is differentiable (.indices() for SparseTensors for example)
                rhs_value = f'as_view({view_info}, {var}, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false)'
            elif len(differentiable_output_vars) == 1:
                # Single differentiable output (Tensor or Tensor[])
                return_info = differentiable_outputs[0]
                # We only support simple Tensor or a TensorList for functions that return views
                if not is_tensor_type(return_info.type) and not is_tensor_list_type(return_info.type):
                    raise RuntimeError(f'{base_name} that return differentiable views can only return Tensor or Tensor[]')
                # Only allow rebasing of the history if we return a single Tensor
                # If we are in a no grad block, raise a warning
                # See NOTE [ View + Inplace detection ] for more details about this logic
                if is_tensor_list_type(return_info.type):
                    if base_name in MULTI_OUTPUT_SAFE_FUNCTIONS:
                        creation_meta = 'CreationMeta::MULTI_OUTPUT_SAFE'
                    else:
                        creation_meta = 'CreationMeta::MULTI_OUTPUT_NODE'
                    call += (f'as_view(/* base */ {view_info}, /* output */ {var}, /* is_bw_differentiable */ true, '
                             '/* is_fw_differentiable */ true, '
                             f'/* creation_meta */ {creation_meta});')
                    rhs_value = f'std::move({var})'
                else:
                    call += emit_view_lambda(unpacked_bindings)
                    creation_meta = 'GradMode::is_enabled() ? CreationMeta::DEFAULT: CreationMeta::NO_GRAD_MODE'
                    rhs_value = (f'as_view(/* base */ {view_info}, /* output */ {var}, /* is_bw_differentiable */ true, '
                                 '/* is_fw_differentiable */ true, '
                                 f'/* view_func */ func, /* creation_meta */ {creation_meta})')
            else:
                # This could be supported but we don't need it at the moment, so keeping things simple.
                raise RuntimeError('Function that return multiple differentiable output '
                                   'when at least one of them is view is not supported.')
        else:
            rhs_value = f'std::move({var})'
        assert rhs_value is not None
        call += ASSIGN_RETURN_VALUE.substitute(return_values=tie_return_values(f),
                                               rhs_value=rhs_value)
        return call

    def enforce_same_tensorimpl_and_storage(call: str, unpacked_bindings: List[Binding]) -> str:
        save_ptrs_stmts: List[str] = []
        enforce_same_ptrs_stmts: List[str] = []
        if cpp.name(f.func) not in DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE:
            for unpacked_binding in unpacked_bindings:
                arg = unpacked_binding.name
                noref_cpp_type = unpacked_binding.ctype.cpp_type(strip_ref=True)
                if noref_cpp_type == 'TensorList':
                    save_ptrs_stmts += [SAVE_TENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                                        SAVE_TENSORLIST_IMPL.substitute(tensorlist_name=arg)]
                    enforce_same_ptrs_stmts += [ENFORCE_SAME_TENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                                                ENFORCE_SAME_TENSORLIST_IMPL.substitute(tensorlist_name=arg)]
                elif noref_cpp_type == 'c10::List<c10::optional<Tensor>>':
                    save_ptrs_stmts += [SAVE_OPTIONALTENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                                        SAVE_OPTIONALTENSORLIST_IMPL.substitute(tensorlist_name=arg)]
                    enforce_same_ptrs_stmts += [ENFORCE_SAME_OPTIONALTENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                                                ENFORCE_SAME_OPTIONALTENSORLIST_IMPL.substitute(tensorlist_name=arg)]
                elif noref_cpp_type == 'Tensor':
                    save_ptrs_stmts += [SAVE_TENSOR_STORAGE.substitute(tensor_name=arg),
                                        SAVE_TENSOR_IMPL.substitute(tensor_name=arg)]
                    enforce_same_ptrs_stmts += [ENFORCE_SAME_TENSOR_STORAGE.substitute(tensor_name=arg),
                                                ENFORCE_SAME_TENSOR_IMPL.substitute(tensor_name=arg)]
        assert (save_ptrs_stmts and enforce_same_ptrs_stmts) or (not save_ptrs_stmts and not enforce_same_ptrs_stmts)
        if save_ptrs_stmts and enforce_same_ptrs_stmts:
            call = RUN_ONLY_IN_DEBUG_MODE.substitute(statements=save_ptrs_stmts) + \
                call + \
                RUN_ONLY_IN_DEBUG_MODE.substitute(statements=enforce_same_ptrs_stmts)
        return call

    def emit_call(f: NativeFunction, unpacked_bindings: List[Binding]) -> str:
        # We only care about adding `at::AutoNonVariableTypeMode` guard for non-variable dispatch
        # (which corresponds to 'use_derived' strategy). The purpose of this guard is to make sure
        # the baseType operations still dispatch to non-Variable type, even if the arguments passed
        # in are now Variables.
        # See NOTE [ Treating Variables as non-Variables in type dispatch ] for details.
        unpacked_args = [b.name for b in unpacked_bindings]
        base_type_call = emit_dispatch_call(f, 'self_', unpacked_args)
        if not modifies_arguments(f) and not returns_void:
            call = DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES.substitute(
                base_type_call=base_type_call)

            call += wrap_output(f, unpacked_bindings, 'tmp')
        else:
            call = DISPATCH_TO_NON_VAR_TYPE_WITHOUT_RETURN_VALUES.substitute(
                base_type_call=base_type_call)
        call = enforce_same_tensorimpl_and_storage(call, unpacked_bindings)
        return call

    def emit_history() -> str:
        fn = 'rebase' if modifies_arguments(f) and view_info is None else 'set'
        output_names = [r.name for r in differentiable_outputs]
        # TODO: flatten allocates a std::vector, which could be expensive
        outs = CodeTemplate("flatten_tensor_args( ${outs} )").substitute(outs=output_names)
        return SET_HISTORY.substitute(fn=fn, differentiable_outputs=outs)

    def emit_save_outputs() -> str:
        if is_out_fn:
            # out functions don't currently support differentiation
            return ''
        if info is not None and info.has_derivatives:
            stmts = save_variables(info.all_saved_outputs, True)
            if len(stmts) == 0:
                return ''
            return CONDITIONAL.substitute(cond='grad_fn', statements=stmts)
        return ''

    def emit_any_requires_grad() -> List[str]:
        return [SETUP_ANY_REQUIRES_GRAD.substitute(
            args_with_derivatives=[arg.name for arg in args_with_derivatives]), ]

    def emit_check_inplace() -> List[str]:
        if not inplace:
            return []
        return [f'check_inplace({arg.name}, _any_requires_grad);' for arg in differentiable_outputs]

    def emit_increment_version(f: NativeFunction) -> List[str]:
        if not modifies_arguments(f):
            return []
        return [f'increment_version({r});' for r in cpp.return_names(f)]

    body: List[str] = []
    unpack_args_stats, unpacked_bindings = unpack_args(f)

    body.extend(unpack_args_stats)
    if requires_derivative:
        body.extend(emit_any_requires_grad())
        body.extend(emit_check_inplace())
        body.extend(setup_derivative(differentiable_inputs))
    body.append(declare_returned_variables(f))

    body.append(emit_call(f, unpacked_bindings))
    body.extend(emit_increment_version(f))
    if requires_derivative:
        # set_flags has to appear after version_counter, because rebase_history
        # requires that the counter is incremented before it is called
        body.append(emit_history())
        body.append(emit_save_outputs())
        body.extend(emit_check_if_in_complex_autograd_allowlist())
    if base_name in RESET_GRAD_ACCUMULATOR:
        # `inplace` implies that there is exactly one output named `self`,
        # so we can keep the generated code easy. If you need to
        # `reset_grad_accumulator` in an operator that's not `inplace`, you can
        # remove this assert but the code generation will get more elaborate
        assert inplace
        body.append('reset_grad_accumulator(self);')
    if not returns_void:
        body.append(f'return {get_return_value(f)};')
    return body

@with_native_function
def unpack_args(f: NativeFunction) -> Tuple[List[str], List[Binding]]:
    body: List[str] = []
    unpacked_bindings: List[Binding] = []

    bindings = [r for a in f.func.schema_order_arguments()
                for r in cpp.argument(a,
                                      method=False,
                                      cpp_no_default_args=set(),
                                      faithful=False,
                                      has_tensor_options=False)]

    for i, binding in enumerate(bindings):
        assert not isinstance(binding.argument, SelfArgument)
        if isinstance(binding.argument, TensorOptionsArguments):
            raise RuntimeError("VariableKernel shouldn't take TensorOptions")

        is_nullable = binding.argument.type.is_nullable()
        if not binding.argument.type.is_tensor_like() or is_nullable:
            unpacked_bindings.append(binding)
            continue

        is_tensor_list = is_tensor_list_type(binding.argument.type)
        ref = (not is_nullable) and not is_tensor_list
        suffix = '_opt' if is_nullable and not is_tensor_list else ''
        body.append(UNPACK_TENSOR.substitute(
            arg_name=binding.name,
            arg_pos=i,
            suffix=suffix,
            ref='&' if ref else '',
        ))
        unpacked_bindings.append(Binding(
            name=binding.name + '_',
            ctype=binding.ctype,
            argument=binding.argument,
            default=binding.default,
        ))

    return body, unpacked_bindings

def dispatch_strategy(fn: NativeFunctionWithDifferentiabilityInfo) -> str:
    """How are we going to call the underlying implementation of a
    declaration?  There are two strategies:

        - use_derived: we want to call the implementation on CPUDoubleType
          (or a similar, derived Type instance).  Because these derived
          instances deal in Tensors, not Variables (it's a completely different
          object, so it doesn't dispatch back to VariableType), code on
          this dispatch path needs to wrap/unwrap tensors.  If the
          derived implementation takes and returns tensors, the
          implementation is usually differentiable (although we also use
          the derived dispatch path for non-differentiable functions
          that we still want to dispatch on the derived Type instance;
          e.g., size())

        - use_type: we want to call the implementation on Type, because
          it is implemented concretely, and the functions it invokes will
          get dispatched back to VariableType (which will ensure that they
          are differentiable.)
    """
    if fn.func.is_abstract or (fn.info is not None and fn.info.has_derivatives):
        # If the function is abstract (not implemented on at::Type), we must
        # call the implementation on the derived type with unpacked tensors.

        # If the function has a derivative specified and is concrete, we could
        # call either implementation. We prefer the calling the derived
        # type's implementation with unpacked tensors because it is more
        # performant in some cases: any internal calls to other ATen functions
        # won't have the history tracked.

        # If the function has a type dispatched argument (i.e. is a factory),
        # we prefer calling the derived type's implementation both because it is
        # more performant and to ensure factory functions return tensors with _version
        # of 0 (probably not strictly necessary, but nice to have to keeps versions simple
        # to understand.

        return 'use_derived'
    else:
        # If the function is concrete (we don't have to override it) and we
        # didn't declare it in derivatives.yaml, we'll assume that it is
        # actually implemented out of differentiable functions. (This
        # assumption might not hold, but then you'll see gradcheck fail.)
        return 'use_type'

def is_tensor_type(t: Type) -> bool:
    # TODO: Should handle optional here?
    return t.is_tensor_like() and t.is_list_like() is None

def is_tensor_list_type(t: Type) -> bool:
    # TODO: Should handle optional here?
    return t.is_tensor_like() and t.is_list_like() is not None

def modifies_arguments(f: NativeFunction) -> bool:
    return f.func.kind() in [SchemaKind.inplace, SchemaKind.out]

def match_differentiability_info(
    native_functions: List[NativeFunction],
    differentiability_infos: Sequence[DifferentiabilityInfo],
) -> List[NativeFunctionWithDifferentiabilityInfo]:
    """Sets the "derivative" key on declarations to matching autograd function

    In-place functions will use the out-of-place derivative definition if there
    is no in-place specific derivative.
    """

    info_by_schema = {info.func.func: info for info in differentiability_infos}
    functional_info_by_signature = {
        info.func.func.signature(strip_default=True): info
        for info in differentiability_infos
        if info.func.func.kind() == SchemaKind.functional}

    def find_info(f: NativeFunction) -> Tuple[Optional[DifferentiabilityInfo], bool]:
        if f.func in info_by_schema:
            return info_by_schema[f.func], True

        # if there is no exact match look for the out-of-place signature.
        # i.e mul() for mul_() or mul_out()
        return functional_info_by_signature.get(f.func.signature(strip_default=True)), False

    result: List[NativeFunctionWithDifferentiabilityInfo] = []
    for f in native_functions:
        info, is_exact_match = find_info(f)
        result.append(NativeFunctionWithDifferentiabilityInfo(
            func=f,
            info=info,
        ))

    return result
