# Generates VariableType.h/cpp
#
# **If any changes are being made to the VariableType codegen please also check
# if updates are needed in torch/csrc/autograd/autograd_not_implemented_fallback.cpp
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
from .context import with_native_function_with_differentiability_info
from .gen_trace_type import (
    MANUAL_BACKEND, MANUAL_AUTOGRAD_AND_TRACER, declare_returned_variables,
    tie_return_values, get_return_value, type_wrapper_name,
)
from .gen_inplace_or_view_type import (
    get_view_info, is_tensor_type, is_tensor_list_type, unpack_args, get_base_name,
    use_derived, modifies_arguments, WRAPPER_REGISTRATION, TMP_VAR, METHOD_DEFINITION,
    ASSIGN_RETURN_VALUE, gen_formals, ALL_VIEW_FUNCTIONS, unpacked_name,
    AUTOGRAD_NOT_IMPLEMENTED_REGISTRATION
)

from tools.codegen.api.types import (Binding, DispatcherSignature, BaseCType, TENSOR_LIST_LIKE_CTYPES, intArrayRefT,
                                     tensorT, tensorListT, iTensorListT, iOptTensorRefListT, MutRefCType, OptionalCType,
                                     ListCType, SpecialArgName, scalarT, stringT,
                                     VectorCType)
from tools.codegen.api.autograd import (
    DifferentiableInput, NativeFunctionWithDifferentiabilityInfo,
    SavedAttribute, dispatch_strategy, gen_differentiable_outputs,
    is_differentiable)
from tools.codegen.api import cpp
from tools.codegen.code_template import CodeTemplate
from tools.codegen.context import native_function_manager, with_native_function
from tools.codegen.utils import mapMaybe, FileManager
from tools.codegen.model import (Argument, NativeFunction, SchemaKind,
                                 SelfArgument, TensorOptionsArguments,
                                 BaseType, ListType)
from typing import Callable, List, Optional, Sequence, Union, Dict

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
    'isnan', 'isposinf', 'isneginf', 'isinf', 'signbit', 'isin',
    # Functions return none are not differentiable
    'record_stream',
    # These functions are not differentiable
    'logical_and', 'logical_xor', 'logical_not', 'logical_or',
}

# The C -> R functions at the time of adding this are still being audited and tested
# but will not error out.
# C -> C, R -> C functions for which backward is correctly implemented and tested
GRADIENT_IMPLEMENTED_FOR_COMPLEX = {
    't', 'view', 'reshape', 'reshape_as', 'view_as', 'roll', 'clone', 'diag_embed',
    'repeat', 'expand', 'flip', 'fliplr', 'flipud', 'rot90', 'transpose',
    'permute', 'squeeze', 'unsqueeze', 'resize', 'resize_as', 'tril',
    'triu', 'chunk', 'zero_', 'eq_', 'ne_', 'add', '__radd__', 'sum',
    '_conj', 'sin', 'cos', 'mul', 'sinc', 'sinh', 'cosh', '__rmul__',
    'sgn', 'asin', 'acos', 'sub', 'div', 'cat', 'view_as_complex', 'index_put',
    'neg', 'complex', 'select', '_s_where', 'as_strided', 'slice', 'constant_pad_nd',
    'unbind', 'split', 'split_with_sizes', 'unsafe_split', 'split_with_sizes_backward',
    'dot', 'vdot', 'cholesky', 'triangular_solve', 'mm', '_unsafe_view', 'mv', 'outer',
    'bmm', 'diagonal', 'alias', 'atan', 'log', 'log10', 'log1p', 'log2', 'reciprocal',
    'tan', 'pow', 'rsqrt', 'tanh', 'tanh_backward', 'asinh', 'acosh', 'atanh', 'take', 'fill_',
    'exp', 'nonzero', 'mean', 'inverse', 'solve', 'linalg_cholesky', 'addcmul', 'addcdiv',
    'matrix_exp', 'linalg_matrix_exp', 'linalg_eigh', 'cholesky_solve', 'linalg_qr', '_linalg_svd', '_fft_c2c', '_fft_r2c',
    'linalg_solve', 'sqrt', 'stack', 'gather', 'index_select', 'index_add_', 'linalg_inv', 'linalg_inv_ex',
    'l1_loss_backward', 'baddbmm', 'addbmm', 'addmm', 'addmv', 'addr', 'linalg_householder_product',
    'constant_pad_nd', 'reflection_pad1d', 'reflection_pad2d', 'reflection_pad3d', 'linalg_cholesky_ex', 'linalg_eig',
    'select_backward', 'diagonal_backward', 'slice_backward',
    'reflection_pad1d_backward', 'reflection_pad2d_backward', 'reflection_pad3d_backward', 'symeig', '_sparse_sparse_matmul',
    'replication_pad1d', 'replication_pad2d', 'replication_pad3d', 'take', 'put_', '_to_copy',
    'replication_pad1d_backward', 'replication_pad2d_backward', 'replication_pad3d_backward',
    'diag', 'masked_scatter', 'masked_select', 'index_add', 'index_fill', 'trace', 'polar', 'cumsum', 'rsub',
    'eig', 'lerp', 'linalg_vector_norm', 'cumprod', 'prod', 'index_copy', 'lu', 'unfold', 'unfold_backward',
    'index', 'masked_fill', 'linalg_cross', 'lu_unpack', 'renorm', '_conj_physical', 'linalg_lu_factor_ex',
    'scatter', 'scatter_add', 'sigmoid', 'sigmoid_backward', 'trapezoid', 'cumulative_trapezoid',
    'conj_physical_', '_neg_view', '_reshape_alias', '_det_lu_based_helper', 'lu_solve',
    'linalg_solve_triangular', 'linalg_pinv', 'linalg_lstsq', 'col2im', 'col2im_backward', 'im2col', 'im2col_backward',
}

GRADIENT_IMPLEMENTED_FOR_SPARSE_COMPLEX = {
    'to_dense', '_coalesce', 'coalesce', 'values', '_sparse_coo_tensor_with_dims_and_tensors',
    'sparse_mask_helper_cuda', '_sparse_addmm',
}

GRADIENT_IMPLEMENTED_FOR_COMPLEX.update(GRADIENT_IMPLEMENTED_FOR_SPARSE_COMPLEX)

# Some operators invalidate the grad_accumulator. Let's reset it.
RESET_GRAD_ACCUMULATOR = {
    'set', 'resize'
}

# NOTE [ TensorImpl and Storage Pointer Sanity Checks ]
#
# We check the following properties:
#   1) A function should never change the input tensors' underlying c10::TensorImpl
#      pointers or c10::Storage pointers, even if it modifies its input tensors (via
#      inplace or out-variants)
# If the function does not modify its arguments, we also check the following properties
# pertaining to its output:
#   2) Its TensorImpl has use_count of 1
#   3) If the function is a view function, it has the same StorageImpl as that of
#      the input it is aliased with. Otherwise, its StorageImpl has use_count of 1
#
# The following code templates implement the checks for this invariant:
SAVE_TENSOR_STORAGE = CodeTemplate("""\
c10::optional<Storage> ${tensor_name}_storage_saved =
  ${tensor_name}.has_storage() ? c10::optional<Storage>(${tensor_name}.storage()) : c10::nullopt;
""")

# If tensor_name == out_tensor_name, used to enforce (1), otherwise used for (2)
ENFORCE_SAME_TENSOR_STORAGE = CodeTemplate("""\
if (${tensor_name}_storage_saved.has_value())
  AT_ASSERT(${tensor_name}_storage_saved.value().is_alias_of(${out_tensor_name}.storage()));
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
for (const auto& tensor : ${tensorlist_name})
  ${tensorlist_name}_storage_saved.push_back(
    tensor.has_value() && tensor->has_storage() ? c10::optional<Storage>(tensor->storage()) : c10::nullopt);
""")

ENFORCE_SAME_OPTIONALTENSORLIST_STORAGE = CodeTemplate("""\
for (size_t i=0; i<${tensorlist_name}.size(); i++) {
  if (${tensorlist_name}_storage_saved[i].has_value())
    AT_ASSERT(${tensorlist_name}_storage_saved[i].value().is_alias_of(
        ${tensorlist_name}[i]->storage()));
}
""")

SAVE_TENSOR_IMPL = CodeTemplate("""\
c10::intrusive_ptr<TensorImpl> ${tensor_name}_impl_saved;
if (${tensor_name}.defined()) ${tensor_name}_impl_saved = ${tensor_name}.getIntrusivePtr();
""")

ENFORCE_SAME_TENSOR_IMPL = CodeTemplate("""\
if (${tensor_name}_impl_saved) AT_ASSERT(${tensor_name}_impl_saved == ${tensor_name}.getIntrusivePtr());
""")

ENFORCE_TENSOR_IMPL_USE_COUNT_LT_OR_EQ_ONE = CodeTemplate("""\
AT_ASSERT(${tensor_name}.use_count() <= 1, "function: ${fn_name}");
""")

ENFORCE_TENSOR_STORAGE_USE_COUNT_EQUALS_ONE = CodeTemplate("""\
if (${tensor_name}.has_storage()) AT_ASSERT(${tensor_name}.storage().use_count() == 1, "function: ${fn_name}");
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
  auto& t = ${tensorlist_name}[i];
  if (t.has_value()) ${tensorlist_name}_impl_saved[i] = t->getIntrusivePtr();
}
""")

ENFORCE_SAME_OPTIONALTENSORLIST_IMPL = CodeTemplate("""\
for (size_t i=0; i<${tensorlist_name}.size(); i++) {
  if (${tensorlist_name}_impl_saved[i])
    AT_ASSERT(${tensorlist_name}_impl_saved[i] == ${tensorlist_name}[i]->getIntrusivePtr());
}
""")

# The following list contains functions that we don't enforce the invariant on.
DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE = {
    # These functions are expected to change impl or storage of input tensors
    'set_', '_cudnn_rnn_flatten_weight',
}
DONT_ENFORCE_TENSOR_IMPL_USE_COUNT = {
    # These non-inplace, non-out functions return tensors with use_count > 1
    # Therefore, they MAY (but not necessarily) return one of its inputs as-is
    # See https://github.com/pytorch/pytorch/issues/60426 for more information
    '_embedding_bag', '_embedding_bag_forward_only',
    'q_per_channel_scales', 'q_per_channel_zero_points',
    'lu_unpack', '_cudnn_rnn_backward',

    # The below failed StorageImpl use_count check but we skip tensor_impl check
    # just in case
    '_cudnn_rnn', 'dequantize_self',
}

DONT_ENFORCE_STORAGE_IMPL_USE_COUNT = {
    # These non-view functions return tensors with storage use_count != 1
    '_slow_conv2d_forward', 'slow_conv3d_forward', 'channel_shuffle',

    # If an input is returned as-is in output, we cannot guarantee its storage_impl
    # use count to be 1 either.
    *DONT_ENFORCE_TENSOR_IMPL_USE_COUNT,
}
# END CHECKS FOR [ TensorImpl and Storage Pointer Sanity Checks ]

DECLARE_GRAD_FN = CodeTemplate("""\
std::shared_ptr<${op}> grad_fn;
""")

SETUP_ANY_REQUIRES_GRAD = CodeTemplate("""\
auto _any_requires_grad = compute_requires_grad( ${args_with_derivatives} );
${extra_differentiability_conditions}
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

CALL_REDISPATCH = CodeTemplate("""\
at::redispatch::${api_name}(${unpacked_args})""")
# If the non-variable operation has return values, we use the `tmp` variable to hold the
# values temporarily and pass the values to the return variables outside of the
# `at::AutoDispatchBelowAutograd` guard block.
DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES = CodeTemplate("""\
auto ${tmp_var} = ([&]() {
  ${guard}
  return ${base_type_call};
})();
""")

DISPATCH_TO_NON_VAR_TYPE_WITHOUT_RETURN_VALUES = CodeTemplate("""\
{
  ${guard}
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

FW_DERIVATIVE_CHECK_TEMPLATE = CodeTemplate("""\
isFwGradDefined(${req_inp})\
""")

FW_DERIVATIVE_DEFINED_GRAD_TEMPLATE = CodeTemplate("""\
auto ${inp}_t_raw = toNonOptFwGrad(${inp});
auto ${inp}_tensor = toNonOptTensor(${inp});
auto ${inp}_t = (${inp}_t_raw.defined() || !${inp}_tensor.defined())
  ? ${inp}_t_raw : at::${zeros_fn}(${inp}_tensor.sizes(), ${inp}_tensor.options());
""")

FW_DERIVATIVE_DEFINED_PRIMAL_TEMPLATE = CodeTemplate("""\
auto ${inp}_p = toNonOptPrimal(${inp});
""")

FW_DERIVATIVE_SETTER_TENSOR = CodeTemplate("""\
if (${out_arg}_new_fw_grad_opt.has_value() && ${out_arg}_new_fw_grad_opt.value().defined()) {
  // The hardcoded 0 here will need to be updated once we support multiple levels.
  ${out_arg}._set_fw_grad(${out_arg}_new_fw_grad_opt.value(), /* level */ 0, /* is_inplace_op */ ${is_inplace});
}
""")

FW_DERIVATIVE_SETTER_TENSOR_LIST = CodeTemplate("""\
if (${out_arg}_new_fw_grad_opt.has_value()) {
  auto ${out_arg}_new_fw_grad = ${out_arg}_new_fw_grad_opt.value();
  TORCH_INTERNAL_ASSERT(${out_arg}.size() == ${out_arg}_new_fw_grad.size());
  for (auto i=0; i<${out_arg}.size(); ++i) {
    if (${out_arg}_new_fw_grad[i].defined()) {
      // The hardcoded 0 here will need to be updated once we support multiple levels.
      ${out_arg}[i]._set_fw_grad(${out_arg}_new_fw_grad[i], /* level */ 0, /* is_inplace_op */ ${is_inplace});
    }
  }
}
""")

FW_DERIVATIVE_TEMPLATE = CodeTemplate("""\
${fw_grad_opt_definition}
if (${requires_fw_grad}) {
    ${unpacked_arguments}
    ${out_arg}_new_fw_grad_opt = ${formula};
}
""")

FW_DERIVATIVE_FORBID_TEMPLATE = CodeTemplate("""\
TORCH_CHECK_NOT_IMPLEMENTED(!(${cond}), "Trying to use forward AD with ${msg} that does not support it.");
""")

FW_DERIVATIVE_FORBID_LIST_TEMPLATE = CodeTemplate("""\
for (const auto& _t: ${arg}) {
    TORCH_CHECK_NOT_IMPLEMENTED(!(${cond}), "Trying to use forward AD with ${msg} that does not support it.");
}
""")

def gen_variable_type(
    out: str,
    native_yaml_path: str,
    fns_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
) -> None:

    """VariableType.h and VariableType.cpp body

    This is the at::Type subclass for differentiable tensors. The
    implementation of each function dispatches to the base tensor type to
    compute the output. The grad_fn is attached to differentiable functions.
    """
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write('VariableType.h', lambda: {
        'generated_comment': "@" f'generated from {template_path}/VariableType.h'
    })

    # NOTE: see Note [Sharded File] at the top of the VariableType.cpp
    # template regarding sharding of the generated files.
    fm.write_sharded(
        'VariableType.cpp',
        [fn for fn in fns_with_diff_infos if use_derived(fn)],
        key_fn=lambda fn: cpp.name(fn.func.func),
        base_env={
            'generated_comment':
            "@" f'generated from {template_path}/VariableType.cpp',
        },
        env_callable=gen_variable_type_func,
        num_shards=5,
        sharded_keys={'type_derived_method_definitions', 'wrapper_registrations'}
    )

@with_native_function
def gen_wrapper_registration(f: NativeFunction) -> str:
    return WRAPPER_REGISTRATION.substitute(
        unqual_operator_name_with_overload=f.func.name,
        type_wrapper_name=type_wrapper_name(f),
        class_type='VariableType',
    )

def gen_variable_type_func(
    fn: NativeFunctionWithDifferentiabilityInfo
) -> Dict[str, List[str]]:
    f = fn.func
    with native_function_manager(f):
        name = cpp.name(f.func)
        formals = gen_formals(f)

        if fn.info is None and not get_base_name(f) in RESET_GRAD_ACCUMULATOR \
                and not get_base_name(f) in DONT_REQUIRE_DERIVATIVE \
                and len(gen_differentiable_outputs(fn)) > 0 \
                and not cpp.name(f.func) in DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE \
                and not type_wrapper_name(f) in DONT_ENFORCE_STORAGE_IMPL_USE_COUNT \
                and not type_wrapper_name(f) in DONT_ENFORCE_TENSOR_IMPL_USE_COUNT:
            # NOTE: [ Registering AutogradNotImplemented boxed kernel ]
            #
            # When there is no derivatives.yaml entry, we register a generic boxed
            # NotImplemented kernel to set grad_fn to be NotImplemented, so that forward
            # proceeds as usual but an error is properly produced on backward.
            # TODO: it would be nice to not have these special cases
            #
            # There are several cases where still let codegen handle it:
            # 1) ops that need to reset grad accumulator (we let codegen handle this case
            #     because) the list is (currently) only accessible in Python.
            # 2) User explicitly specifies DONT_REQUIRE_DERIVATIVE. This basically makes
            #    autograd a fallthrough with NDEBUG checks. This can be useful for when all
            #    outputs are integral.
            # 3) When there are no differentiable outputs. This is similar to (2).
            # 4) There are certain ops where we skip certain NDEBUG checks. this is similar
            #    to (1).
            type_definition = ""
            wrapper_registration = AUTOGRAD_NOT_IMPLEMENTED_REGISTRATION.substitute(
                unqual_operator_name_with_overload=f.func.name)
        else:
            type_definition = METHOD_DEFINITION.substitute(
                return_type=cpp.returns_type(f.func.returns).cpp_type(),
                type_wrapper_name=type_wrapper_name(f),
                type_definition_body=emit_body(fn),
                formals=formals,
            )
            wrapper_registration = gen_wrapper_registration(f)

    # See Note [Manual Backend kernels]
    assert (name in MANUAL_BACKEND) == f.manual_kernel_registration
    # If you want to register a kernel to Autograd, you must make the op abstract.
    # In other words, this op must have dispatch section in native_functions.yaml.
    if name in MANUAL_AUTOGRAD_AND_TRACER or (fn.info and fn.info.has_derivatives):
        msg = (f'There\'s a formula for {name}(or its functional variant) in derivatives.yaml. '
               f'It\'s required to add a dispatch section for it with explicit supported backends e.g CPU/CUDA '
               f'or CompositeExplicitAutograd in native_functions.yaml. Please see '
               f'https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native#choosing-the-right-dispatch-keyword '
               f'for instructions to choose the right dispatch keyword.')
        assert f.is_abstract, msg

    return {
        'type_derived_method_definitions': [type_definition],
        'wrapper_registrations': [wrapper_registration],
    }

@with_native_function_with_differentiability_info
def emit_body(fn: NativeFunctionWithDifferentiabilityInfo) -> List[str]:
    assert dispatch_strategy(fn) == 'use_derived'
    f = fn.func
    info = fn.info
    fw_derivatives = fn.fw_derivatives

    name = cpp.name(f.func)
    inplace = f.func.kind() == SchemaKind.inplace
    is_out_fn = f.func.kind() == SchemaKind.out
    returns_void = len(f.func.returns) == 0
    base_name = get_base_name(f)
    view_info = get_view_info(f)

    def gen_differentiable_input(
        arg: Union[Argument, SelfArgument, TensorOptionsArguments]
    ) -> Optional[DifferentiableInput]:
        if isinstance(arg, TensorOptionsArguments):
            return None
        a: Argument = arg.argument if isinstance(arg, SelfArgument) else arg

        # TODO: `cpp_type` is only to keep it byte-for-byte compatible with the old codegen, should remove.
        # NB: This is not a clone of cpp.argument() - TensorOptionsArguments / faithful / binds are
        # not handled properly as they are irrelevant for this codegen.
        cpp_type = cpp.argument_type(a, binds=a.name, structured_type_override=f.part_of_structured_group).cpp_type()

        if not is_differentiable(a.name, a.type, info):
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

    differentiable_inputs = gen_differentiable_inputs(f)
    args_with_derivatives = find_args_with_derivatives(differentiable_inputs)
    differentiable_outputs = gen_differentiable_outputs(fn)

    undifferentiable = (base_name in DONT_REQUIRE_DERIVATIVE) or (name in DONT_REQUIRE_DERIVATIVE)

    requires_derivative = (not undifferentiable) and (len(differentiable_inputs) > 0) and (len(differentiable_outputs) > 0)

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
            if arg.nctype.type != BaseCType(tensorT):
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
            if arg.cpp_type == 'at::Tensor' or arg.cpp_type in TENSOR_LIST_LIKE_CTYPES:
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
            arg_name = arg.name
            if info and arg_name in info.non_differentiable_arg_names:
                continue
            if arg_name == 'output':
                # Double-backwards definitions sometimes take in 'input' and
                # 'output', but only define the derivative for input.
                continue
            body.append(f'check_no_requires_grad({arg_name}, "{arg_name}", "{name}");')
        return body

    def emit_original_self_definition() -> List[str]:
        body: List[str] = []
        if inplace:
            body.append('c10::optional<at::Tensor> original_self;')

            all_forward_grad_cond = []
            for derivative in fw_derivatives:
                if derivative.required_original_self_value:
                    all_forward_grad_cond.append(get_any_has_forward_grad_name(derivative.var_name))

            if all_forward_grad_cond:
                body.append(f'if ({" || ".join(all_forward_grad_cond)}) {{')
                body.append('  original_self = self.clone();')
                body.append('}')

        return body

    def save_variables(
        saved_variables: Sequence[SavedAttribute],
        is_output: bool,
        guard_for: Callable[[SavedAttribute], Optional[str]] = lambda name: None,
    ) -> Sequence[str]:
        # assign the saved variables to the generated grad_fn
        stmts: List[str] = []
        for arg in saved_variables:
            name = arg.nctype.name.name if isinstance(arg.nctype.name, SpecialArgName) else arg.nctype.name
            type = arg.nctype.type
            expr = arg.expr
            stmts_prepend = None
            if type == BaseCType(tensorT) or type == OptionalCType(BaseCType(tensorT)) or \
                    type == MutRefCType(OptionalCType(BaseCType(tensorT))) or (is_output and type == BaseCType(scalarT)):
                var = name
                name += '_'
                if var == 'self' and inplace:
                    stmts_prepend = 'if (!original_self.has_value()) original_self = self.clone()'
                    var = 'original_self.value()'
                    assert not is_output
                if inplace and is_output:
                    var = 'self'
                    is_inplace_view = f'{var}.is_view()'
                    expr = f'SavedVariable({var}, {str(is_output).lower()}, {is_inplace_view})'
                else:
                    expr = f'SavedVariable({var}, {str(is_output).lower()})'
            elif type == BaseCType(tensorListT) or type == ListCType(OptionalCType(BaseCType(tensorT))) or \
                 type == BaseCType(iTensorListT) or type == BaseCType(iOptTensorRefListT):
                expr = f'make_saved_variable_list({name})'
                name += '_'
            elif type == BaseCType(intArrayRefT):
                expr = expr + ".vec()"
            elif type == BaseCType(stringT):
                expr = f'std::string({expr})'
            elif type == OptionalCType(BaseCType(stringT)):
                expr = f'{expr}.has_value() ? c10::optional<std::string>(std::string({expr}.value())) : c10::nullopt'
            guard = guard_for(arg)
            if guard is None:
                if stmts_prepend:
                    stmts.append(f'{stmts_prepend};')
                stmts.append(f'grad_fn->{name} = {expr};')
            else:
                stmts.append(f'if ({guard}) {{')
                if stmts_prepend:
                    stmts.append(f'  {stmts_prepend};')
                stmts.append(f'  grad_fn->{name} = {expr};')
                stmts.append('}')
        return stmts

    # Generates a Dispatcher::redispatch() call into the dispatcher. We do this mainly for performance reasons:
    #  - Pre-compute the full DispatchKeySet. This saves the dispatcher from having to read from TLS.
    #  - redispatch() avoids a redundant call to RecordFunction, which was already called right before
    #    we entered this autograd kernel.
    def emit_dispatch_call(f: NativeFunction, input_base: str, unpacked_args: Sequence[str]) -> str:
        """ Dispatch call via function in a namespace or method on Tensor."""
        dispatcher_sig = DispatcherSignature.from_schema(f.func, structured_type_override=f.part_of_structured_group)
        dispatcher_exprs = dispatcher_sig.exprs()

        # code-generated autograd kernels plumb and recompute dispatch keys directly through the kernel for performance.
        # Ops also always have a function variant of the redispatch API.
        # See Note [Plumbing Keys Through The Dispatcher] for details.
        dispatch_key_set = 'ks & c10::after_autograd_keyset'
        call = CALL_REDISPATCH.substitute(
            api_name=cpp.name(
                f.func,
                faithful_name_for_out_overloads=True,
            ),
            unpacked_args=[dispatch_key_set] + list(unpacked_args))
        return call

    def wrap_output(f: NativeFunction, unpacked_bindings: List[Binding], var: str) -> str:
        call = ''
        rhs_value: Optional[str] = None
        if not any(r.type.is_tensor_like() for r in f.func.returns):
            rhs_value = var
        else:
            rhs_value = f'std::move({var})'
        assert rhs_value is not None
        call += ASSIGN_RETURN_VALUE.substitute(return_values=tie_return_values(f),
                                               rhs_value=rhs_value)
        return call

    def check_tensorimpl_and_storage(call: str, unpacked_bindings: List[Binding]) -> str:
        # See NOTE [ TensorImpl and Storage Pointer Sanity Checks ]
        stmts_before_call: List[str] = []
        stmts_after_call: List[str] = []

        if cpp.name(f.func) in DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE:
            return call

        # Check properties of inputs (enforce (1))
        for unpacked_binding in unpacked_bindings:
            arg = unpacked_binding.name
            noref_cpp_type = unpacked_binding.nctype.type.remove_const_ref()
            if noref_cpp_type == BaseCType(tensorListT) or noref_cpp_type == BaseCType(iTensorListT):
                stmts_before_call += [SAVE_TENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                                      SAVE_TENSORLIST_IMPL.substitute(tensorlist_name=arg)]
                stmts_after_call += [ENFORCE_SAME_TENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                                     ENFORCE_SAME_TENSORLIST_IMPL.substitute(tensorlist_name=arg)]
            elif noref_cpp_type == ListCType(OptionalCType(BaseCType(tensorT))) or noref_cpp_type == BaseCType(iOptTensorRefListT):
                stmts_before_call += [SAVE_OPTIONALTENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                                      SAVE_OPTIONALTENSORLIST_IMPL.substitute(tensorlist_name=arg)]
                stmts_after_call += [ENFORCE_SAME_OPTIONALTENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                                     ENFORCE_SAME_OPTIONALTENSORLIST_IMPL.substitute(tensorlist_name=arg)]
            elif noref_cpp_type == BaseCType(tensorT):
                stmts_before_call += [SAVE_TENSOR_STORAGE.substitute(tensor_name=arg),
                                      SAVE_TENSOR_IMPL.substitute(tensor_name=arg)]
                stmts_after_call += [ENFORCE_SAME_TENSOR_STORAGE.substitute(tensor_name=arg, out_tensor_name=arg),
                                     ENFORCE_SAME_TENSOR_IMPL.substitute(tensor_name=arg)]

        assert (stmts_before_call and stmts_after_call) or (not stmts_before_call and not stmts_after_call)

        # Check properties of outputs (enforce (2), (3))
        if not f.func.kind() in (SchemaKind.inplace, SchemaKind.out):
            base_name = f.func.name.name.base  # TODO: should be str(f.func.name.name)?
            aliased_arg_name = ALL_VIEW_FUNCTIONS.get(base_name, None)
            if aliased_arg_name is not None:
                aliased_arg_name = unpacked_name(aliased_arg_name)
            for i, (ret, ret_name) in enumerate(zip(f.func.returns, cpp.return_names(f))):
                noref_cpp_type = cpp.return_type(ret).remove_const_ref()
                if noref_cpp_type == BaseCType(tensorT):
                    if aliased_arg_name is not None:
                        assert i == 0, "Expect non-CompositeImplicitAutograd view function {base} to return single output"
                        stmts_after_call += [ENFORCE_SAME_TENSOR_STORAGE.substitute(tensor_name=aliased_arg_name,
                                                                                    out_tensor_name=ret_name)]
                    else:
                        if type_wrapper_name(f) not in DONT_ENFORCE_STORAGE_IMPL_USE_COUNT:
                            stmts_after_call += [ENFORCE_TENSOR_STORAGE_USE_COUNT_EQUALS_ONE.substitute(
                                tensor_name=ret_name, fn_name=type_wrapper_name(f))]

                    if type_wrapper_name(f) not in DONT_ENFORCE_TENSOR_IMPL_USE_COUNT:
                        stmts_after_call += [ENFORCE_TENSOR_IMPL_USE_COUNT_LT_OR_EQ_ONE.substitute(
                            tensor_name=ret_name, fn_name=type_wrapper_name(f))]

                # Currently we don't have any functions that return the following types, but
                # we should update the checks once we do
                elif noref_cpp_type == ListCType(OptionalCType(BaseCType(tensorT))):
                    raise AssertionError(f"Please add use_count checks for {noref_cpp_type}")
                elif noref_cpp_type == BaseCType(tensorListT):
                    raise AssertionError(f"Please add use_count checks for {noref_cpp_type}")

        if stmts_before_call and stmts_after_call:
            call = RUN_ONLY_IN_DEBUG_MODE.substitute(statements=stmts_before_call) + \
                call + \
                RUN_ONLY_IN_DEBUG_MODE.substitute(statements=stmts_after_call)
        return call

    def emit_call(f: NativeFunction, unpacked_bindings: List[Binding]) -> str:
        # We only care about adding `at::AutoDispatchBelowAutograd` guard for non-variable dispatch
        # (which corresponds to 'use_derived' strategy). The purpose of this guard is to make sure
        # the baseType operations still dispatch to non-Variable type, even if the arguments passed
        # in are now Variables.
        # See NOTE [ Treating Variables as non-Variables in type dispatch ] for details.
        unpacked_args = [b.name for b in unpacked_bindings]
        base_type_call = emit_dispatch_call(f, 'self_', unpacked_args)

        if get_view_info(f) is not None or modifies_arguments(f):
            guard = 'at::AutoDispatchBelowAutograd guard;'
        else:
            guard = 'at::AutoDispatchBelowADInplaceOrView guard;'

        if not modifies_arguments(f) and not returns_void:
            call = DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES.substitute(
                base_type_call=base_type_call, tmp_var=TMP_VAR, guard=guard)

            call += wrap_output(f, unpacked_bindings, TMP_VAR)
        else:
            call = DISPATCH_TO_NON_VAR_TYPE_WITHOUT_RETURN_VALUES.substitute(
                base_type_call=base_type_call, guard=guard)
        call = check_tensorimpl_and_storage(call, unpacked_bindings)
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
        extra_condition = ''
        if fn.info and fn.info.output_differentiability_conditions:
            assert len(fn.info.output_differentiability_conditions) == 1
            extra_condition = \
                f'_any_requires_grad &= ({fn.info.output_differentiability_conditions[0]});'
        return [SETUP_ANY_REQUIRES_GRAD.substitute(
            args_with_derivatives=[arg.name for arg in args_with_derivatives],
            extra_differentiability_conditions=extra_condition)]

    def get_any_has_forward_grad_name(var_name: str) -> str:
        return f'_any_has_forward_grad_{var_name}'

    def emit_any_has_forward_grad() -> List[str]:
        content: List[str] = []
        for derivative in fw_derivatives:
            assert derivative.required_inputs_fw_grad is not None
            requires_fw_grad = " || ".join([FW_DERIVATIVE_CHECK_TEMPLATE.substitute(req_inp=inp.name)
                                           for inp in differentiable_inputs if inp.name in derivative.required_inputs_fw_grad])
            if not requires_fw_grad:
                # Handle functions like stack
                # For these, we don't unpack anything and always call the user function
                if not (len(differentiable_inputs) == 1 and is_tensor_list_type(differentiable_inputs[0].type)):
                    raise RuntimeError(f'No differentiable input to "{name}" is a differentiable Tensor (as the provided'
                                       'forward AD formula does not use any input tangent) even though a forward gradient '
                                       'formula has been defined for it. This case should only happen for function that '
                                       'take a single TensorList as input. All other cases are not supported right now.')
                requires_fw_grad = "true"

            if fn.info and fn.info.output_differentiability_conditions:
                assert len(fn.info.output_differentiability_conditions) == 1
                requires_fw_grad = \
                    f'({fn.info.output_differentiability_conditions[0]}) && ({requires_fw_grad})'

            content.append(f"auto {get_any_has_forward_grad_name(derivative.var_name)} = {requires_fw_grad};\n"
                           f"(void){get_any_has_forward_grad_name(derivative.var_name)};")

        return content

    def emit_check_inplace() -> List[str]:
        if not inplace:
            return []
        return [f'check_inplace({arg.name}, _any_requires_grad);' for arg in differentiable_outputs]

    def emit_fw_derivatives() -> List[str]:
        content: List[str] = []
        fw_grad_setters: List[str] = []
        for derivative in fw_derivatives:
            res = derivative.var_name
            if f.func.name.name.inplace:
                # TODO update this when inplace namings are unified
                res = "self"

            assert derivative.required_inputs_fw_grad is not None

            unpacked_arguments = ""
            for inp in differentiable_inputs:
                zeros_fn = "zeros" if inplace and inp.name == "self" else "_efficientzerotensor"
                if inp.name in derivative.required_inputs_fw_grad:
                    unpacked_arguments += FW_DERIVATIVE_DEFINED_GRAD_TEMPLATE.substitute(inp=inp.name, zeros_fn=zeros_fn)
                if inp.name in (derivative.required_inputs_primal or []):
                    unpacked_arguments += FW_DERIVATIVE_DEFINED_PRIMAL_TEMPLATE.substitute(inp=inp.name)
            if derivative.required_original_self_value:
                unpacked_arguments += FW_DERIVATIVE_DEFINED_GRAD_TEMPLATE.substitute(inp="original_self", zeros_fn=zeros_fn)
                unpacked_arguments += FW_DERIVATIVE_DEFINED_PRIMAL_TEMPLATE.substitute(inp="original_self")
            elif inplace and derivative.is_reusing_outplace_formula:
                # The gradient wasn't already cloned, do it if grad mode is enabled
                unpacked_arguments += "self_t = GradMode::is_enabled() ? self_t.clone() : self_t;"

            if inplace:
                is_inplace_str = "true"
            else:
                is_inplace_str = "false"

            if isinstance(derivative.var_type, BaseType) and derivative.var_type.is_tensor_like():
                # Is there a way to get from BaseType to BaseCType
                opt_res_grad_type = OptionalCType(BaseCType(tensorT)).cpp_type()
                fw_grad_setter = FW_DERIVATIVE_SETTER_TENSOR.substitute(out_arg=res, is_inplace=is_inplace_str)
            elif isinstance(derivative.var_type, ListType) and derivative.var_type.is_tensor_like():
                opt_res_grad_type = OptionalCType(VectorCType(BaseCType(tensorT))).cpp_type()
                fw_grad_setter = FW_DERIVATIVE_SETTER_TENSOR_LIST.substitute(out_arg=res, is_inplace=is_inplace_str)
            else:
                raise RuntimeError("Unsupported output type for forward derivative")

            fw_grad_opt_definition = f"{opt_res_grad_type} {res}_new_fw_grad_opt = c10::nullopt;"

            # View ops create fw_grad that already is a view of the base's fw_grad so just use that
            content.append(FW_DERIVATIVE_TEMPLATE.substitute(
                fw_grad_opt_definition=fw_grad_opt_definition,
                requires_fw_grad=get_any_has_forward_grad_name(derivative.var_name), formula=derivative.formula, out_arg=res,
                unpacked_arguments=unpacked_arguments))
            fw_grad_setters.append(fw_grad_setter)

        # Set all the grads at the end to avoid: https://github.com/pytorch/pytorch/issues/67367
        content.append('\n'.join(fw_grad_setters))
        return content

    def emit_forbid_fw_derivatives(is_out_fn: bool = False) -> str:
        def get_msg() -> str:
            if is_out_fn:
                msg = name + " (because it is an out= function)"
            else:
                msg = name
            return msg
        res = ""
        to_check: List[str] = []
        for inp in list(mapMaybe(gen_differentiable_input,
                                 f.func.arguments.non_out + list(f.func.arguments.out))):  # type: ignore[operator]
            if is_tensor_type(inp.type):
                to_check.append(FW_DERIVATIVE_CHECK_TEMPLATE.substitute(req_inp=inp.name))
            elif is_tensor_list_type(inp.type):
                cond = FW_DERIVATIVE_CHECK_TEMPLATE.substitute(req_inp="_t")
                res += FW_DERIVATIVE_FORBID_LIST_TEMPLATE.substitute(arg=inp.name, cond=cond, msg=get_msg())
            else:
                raise RuntimeError(f'Unsupported input type for "{name}" when forbidding forward AD usage.')

        if len(to_check) > 0:
            cond = " || ".join(to_check)
            res += FW_DERIVATIVE_FORBID_TEMPLATE.substitute(cond=cond, msg=get_msg())
        return res

    body: List[str] = []
    unpack_args_stats, unpacked_bindings = unpack_args(f)

    body.extend(unpack_args_stats)
    if requires_derivative:
        body.extend(emit_any_requires_grad())
        body.extend(emit_any_has_forward_grad())
        body.extend(emit_check_inplace())
        body.extend(emit_original_self_definition())
        body.extend(setup_derivative(differentiable_inputs))
    body.append(declare_returned_variables(f))

    body.append(emit_call(f, unpacked_bindings))
    if requires_derivative:
        # set_flags has to appear after version_counter, because rebase_history
        # requires that the counter is incremented before it is called
        body.append(emit_history())
        body.extend(emit_check_if_in_complex_autograd_allowlist())

    if is_out_fn:
        body.append(emit_forbid_fw_derivatives(is_out_fn=True))
    else:
        if requires_derivative:
            body.extend(emit_fw_derivatives())
            if len(fw_derivatives) == 0:
                body.append(emit_forbid_fw_derivatives())
            else:
                assert len(fw_derivatives) == len(differentiable_outputs), (
                    "Expected the number of forward derivatives implemented to match the "
                    "number of differentiable outputs. NB: This only applies when at least "
                    "one forward derivative is implemented. Not implementing any forward "
                    "derivatives is also okay, and we would require inputs to the op to "
                    "not have associated tangents in that case.")

    if requires_derivative:
        # Save only after the forward AD has been set up
        body.append(emit_save_outputs())

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
