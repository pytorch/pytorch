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

from .utils import CodeTemplate, nested_dict, write, make_out_api_name_faithful
from .gen_autograd import VIEW_FUNCTIONS, VIEW_FUNCTIONS_WITH_METADATA_CHANGE, \
    MULTI_OUTPUT_SAFE_FUNCTIONS, RETURNS_VIEWS_OF_INPUT
from .gen_autograd_functions import uses_single_grad
from .gen_trace_type import MANUAL_BACKEND, MANUAL_AUTOGRAD_AND_TRACER, MANUAL_AUTOGRAD

from tools.codegen.api.types import *
from tools.codegen.api.autograd import *
import tools.codegen.api.cpp as cpp
import tools.codegen.api.python as python
from tools.codegen.gen import with_native_function
from tools.codegen.model import *
from typing import Dict, Optional, List, Sequence, Any, Callable

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
    'permute', 'squeeze', 'unsqueeze', 'resize', 'resize_as', 'tril', 'triu',
    'chunk', 'split', 'split_with_sizes', 'repeat', 'expand', 'zero_', 'eq_',
    'ne_', 'add', '__radd__', 'sum', '_conj', 'sin', 'cos', 'mul', 'sinc', 'sinh',
    'cosh', '__rmul__', 'sgn', 'asin', 'acos', 'sub', 'div', 'cat', 'view_as_complex',
    'neg', 'complex', 'select', '_s_where', 'as_strided', 'slice', 'constant_pad_nd',
    'unbind', 'split', 'split_with_sizes', 'unsafe_split', 'split_with_sizes_backward',
    'dot', 'vdot', 'cholesky', 'triangular_solve', 'mm', '_unsafe_view', 'mv', 'ger',
    'bmm', 'diagonal', 'alias', 'atan', 'log', 'log10', 'log1p', 'log2', 'reciprocal',
    'tan', 'pow', 'rsqrt', 'tanh', 'tanh_backward', 'asinh', 'acosh', 'take', 'fill_',
    'exp', 'nonzero', 'mean', 'inverse', 'solve', 'linalg_cholesky', 'addcmul', 'addcdiv',
    'matrix_exp', 'linalg_eigh', 'cholesky_solve', 'linalg_qr', 'svd', '_fft_c2c', '_fft_r2c',
    'linalg_solve', 'sqrt', 'stack', 'gather', 'index_select', 'index_add_'
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

# NOTE[UnboxedOnly] Many of our codegen templates currently exist twice, once
# in an _UNBOXEDONLY_ variant and once without _UNBOXEDONLY_. This is because
# ops that are `use_c10_dispatcher: full` need different c++ code than ops
# that aren't `use_c10_dispatcher: full` yet. The _UNBOXEDONLY_ variants
# are for ops that aren't `use_c10_dispatcher: full` yet and those code templates
# can be deleted once all ops are `use_c10_dispatcher: full`.
# If you update one of the templates, you likely also have to update the other.

# See NOTE[UnboxedOnly]
UNBOXEDONLY_WRAPPER_REGISTRATION = CodeTemplate("""\
m.impl_UNBOXED("${unqual_operator_name_with_overload}", &${class_type}::${type_wrapper_name});
""")

WRAPPER_REGISTRATION = CodeTemplate("""\
m.impl("${unqual_operator_name_with_overload}",
       TORCH_FN(${class_type}::${type_wrapper_name})
);
""")

UNPACK_TENSOR = CodeTemplate("""\
auto${ref} ${arg_name}_ = unpack${suffix}(${arg_name}, "${arg_name}", ${arg_pos});""")

LEGACY_WRAP_OPTIONS = CodeTemplate("""\
auto ${arg_name}_ = TensorOptions(${arg_name});""")

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

# Methods shared by TraceType and VariableType to handle return variable declaration, tie and tuple.
def format_return_variables(declaration):
    name = declaration['name']
    arguments = declaration['arguments']
    inplace = declaration['inplace']
    is_out_fn = name.endswith('_out')
    modifies_arguments = inplace or is_out_fn

    def declare_returned_variables():
        if modifies_arguments:
            return ''
        if len(declaration['returns']) == 1:
            return ''
        # TODO: this will be ugly
        names = [ret['type'] + ' ' + ret['name'] + ';' for ret in declaration['returns']]
        return '\n'.join(names)

    def tie_return_values():
        if len(declaration['returns']) == 1:
            return 'auto {}'.format(declaration['returns'][0]['name'])
        names = [ret['name'] for ret in declaration['returns']]
        return 'std::tie({})'.format(', '.join(names))

    def get_return_value():
        if inplace:
            return 'self'
        if is_out_fn:
            return_names = [arg['name'] for arg in arguments
                            if arg.get('output', False)]
            if len(return_names) == 1:
                return return_names[0]
            return 'std::forward_as_tuple({})'.format(', '.join(return_names))

        returns = declaration['returns']
        if len(returns) == 1:
            return returns[0]['name']
        moved = ['std::move({})'.format(r['name']) for r in returns]
        return 'std::make_tuple({})'.format(', '.join(moved))

    return (declare_returned_variables(), tie_return_values(), get_return_value())


def gen_variable_type(out, aten_declarations, differentiability_infos, template_path):

    """VariableType.h and VariableType.cpp body

    This is the at::Type subclass for differentiable tensors. The
    implementation of each function dispatches to the base tensor type to
    compute the output. The grad_fn is attached to differentiable functions.
    """

    aten_declarations = list(sorted(aten_declarations, key=lambda decl: decl['name']))
    match_declarations_with_differentiability_info(aten_declarations, differentiability_infos)

    gen_variable_type_shard(out, aten_declarations, template_path, None, True)

    # NOTE: see Note [Sharded File] at the top of the VariableType.cpp
    # template regarding sharding of the generated files.
    num_shards = 5
    shards = [[] for _ in range(num_shards)]

    # functions are assigned arbitrarily but stably to a file based on hash
    for decl in aten_declarations:
        x = sum(ord(c) for c in decl['name']) % num_shards
        shards[x].append(decl)

    for i, shard in enumerate(shards):
        gen_variable_type_shard(out, shard, template_path, '_%d' % i, False)
    gen_variable_type_shard(out, aten_declarations, template_path, 'Everything', False)


def gen_variable_type_shard(out, aten_declarations, template_path, suffix, header):
    VARIABLE_TYPE_H = CodeTemplate.from_file(template_path + '/VariableType.h')
    VARIABLE_TYPE_CPP = CodeTemplate.from_file(template_path + '/VariableType.cpp')

    type_declarations = []
    type_definitions = []
    wrapper_registrations = []

    for declaration in aten_declarations:
        if declaration['use_c10_dispatcher'] in ['full', 'hacky_wrapper_for_legacy_signatures']:
            formals = declaration['schema_order_formals']
        else:
            assert declaration['use_c10_dispatcher'] == 'with_codegenerated_unboxing_wrapper'
            formals = declaration['formals']
        type_declarations.append(METHOD_DECLARATION.substitute(declaration, formals=formals))
        strategy = dispatch_strategy(declaration)
        if declaration['name'] not in MANUAL_AUTOGRAD and strategy == 'use_derived':
            body = emit_body(declaration)

            type_definitions.append(METHOD_DEFINITION.substitute(
                declaration, type_definition_body=body, formals=formals))
            if declaration['use_c10_dispatcher'] in ['full', 'hacky_wrapper_for_legacy_signatures']:
                wrapper_registrations.append(WRAPPER_REGISTRATION.substitute(
                    declaration, class_type='VariableType'))
            else:
                assert declaration['use_c10_dispatcher'] == 'with_codegenerated_unboxing_wrapper'
                wrapper_registrations.append(UNBOXEDONLY_WRAPPER_REGISTRATION.substitute(
                    declaration, class_type='VariableType'))

        # See Note [Manual Backend kernels]
        assert (declaration['name'] in MANUAL_BACKEND) == declaration['manual_kernel_registration']
        # If you want to register a kernel to Autograd, you must make the op abstract.
        # In other words, this op must have dispatch section in native_functions.yaml.
        if declaration['name'] in MANUAL_AUTOGRAD_AND_TRACER or declaration['derivative']:
            msg = (f'There\'s a formula for {declaration["name"]}(or its functional variant) in derivatives.yaml. '
                   f'It\'s required to add a dispatch section for it with explicit supported backends e.g CPU/CUDA '
                   f'or DefaultBackend in native_functions.yaml. Please see '
                   f'https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native#choosing-the-right-dispatch-keyword '
                   f'for instructions to choose the right dispatch keyword.')
            assert declaration['abstract'], msg

    env = {
        'type_derived_method_declarations': type_declarations,
        'type_derived_method_definitions': type_definitions,
        'wrapper_registrations': wrapper_registrations,
    }
    if header:
        write(out, 'VariableType.h', VARIABLE_TYPE_H, env)
    else:
        write(out, 'VariableType%s.cpp' % suffix, VARIABLE_TYPE_CPP, env)


def emit_body(declaration):
    assert dispatch_strategy(declaration) == 'use_derived'

    arguments = declaration['arguments']
    returns = declaration['returns']
    func = declaration['derivative']
    name = declaration['name']
    inplace = declaration['inplace']
    is_out_fn = name.endswith('_out')
    modifies_arguments = inplace or is_out_fn
    returns_void = len(returns) == 0

    base_name = name[:-1] if inplace else name[:-4] if is_out_fn else name
    view_info = VIEW_FUNCTIONS.get(base_name, None)
    if view_info is None and base_name in RETURNS_VIEWS_OF_INPUT:
        view_info = "self"

    def is_differentiable(arg):
        if 'TensorOptions' in arg['type']:
            return False
        if 'Tensor' not in arg['type']:
            return False
        if arg['name'] in declaration.get('non_differentiable_arg_names', []):
            return False
        return True

    def find_args_with_derivatives(differentiable_inputs):
        """Find arguments that have derivative definitions"""
        if func is None:
            return differentiable_inputs
        names = set(name for d in func.derivatives for name in d.var_names)
        differentiable = [arg for arg in differentiable_inputs if arg['name'] in names]
        if len(differentiable) != len(names):
            missing = names - set(arg['name'] for arg in differentiable)
            raise RuntimeError(f'Missing arguments for derivatives: {missing} in {func.name}')
        return differentiable

    inputs = [arg for arg in arguments if not arg.get('output', False)]
    differentiable_inputs = list(filter(is_differentiable, inputs))
    args_with_derivatives = find_args_with_derivatives(differentiable_inputs)
    non_differentiable_arg_names = declaration.get('non_differentiable_arg_names', [])
    candidate_differentiable_outputs = list(filter(is_differentiable, returns))

    if declaration['output_differentiability'] is not None:
        differentiable_outputs = []
        output_differentiability = declaration['output_differentiability']
        if False in output_differentiability and inplace:
            raise RuntimeError("output_differentiability=False for inplace operation (version_counter won't get updated)")
        for differentiable, output in zip(output_differentiability, returns):
            if differentiable:
                differentiable_outputs.append(output)
    elif uses_single_grad(func):
        differentiable_outputs = candidate_differentiable_outputs[:1]
    else:
        differentiable_outputs = candidate_differentiable_outputs

    requires_derivative = (
        base_name not in DONT_REQUIRE_DERIVATIVE and name not in DONT_REQUIRE_DERIVATIVE and
        len(differentiable_inputs) > 0 and len(differentiable_outputs) > 0)

    if func is not None and not requires_derivative:
        raise RuntimeError('ERROR: derivative ignored for {} -- specified an autograd function without derivative'
                           .format(name))

    def emit_save_inputs():
        setup = []
        if func is None:
            return setup

        has_tensorlist_arg = \
            any(arg.type in ['TensorList', 'const c10::List<c10::optional<Tensor>> &'] for arg in func.args_with_derivatives)

        # We don't want to save tensors if we know that they will never be used
        # when computing the derivative, so we add guards to those statements
        def guard_for(arg: SavedAttribute) -> Optional[str]:
            # It's hard to determine the edge offset if we have TensorLists
            if has_tensorlist_arg:
                return None

            # Empirical evaluation of the cases where we insert those guards in
            # backward show that they are somewhat useless. E.g. there's no need
            # to guard on some values captured from forward, because they had to
            # require_grad if the backward function even gets executed. I don't
            # have any good ideas for detecting those cases, so I simply disabled the
            # checks.
            if 'backward' in func.name:
                return None

            # If there's a single derivative we could compute, we already have
            # a requires_grad check that is sufficient
            if len(func.args_with_derivatives) <= 1:
                return None

            # We really only care about trimming down the amount of tensors we save
            if arg.type != 'Tensor':
                return None

            # We want to emit simple guards, so we only allow that if checking one
            # input is enough to determine whether we need that value
            used_in = [d for d in func.derivatives if arg in d.saved_inputs]
            assert len(used_in) > 0
            if len(used_in) != 1:
                return None
            derivative = used_in[0]
            if len(derivative.var_names) != 1:
                return None
            derivative_var_name = derivative.var_names[0]

            # Figure out the offset of the edge that uses this variable
            for edge_off, arg in enumerate(func.args_with_derivatives):
                if arg.name == derivative_var_name:
                    break
            else:
                raise AssertionError()

            return f'grad_fn->should_compute_output({edge_off})'

        setup.extend(save_variables(func.all_saved_inputs, False, guard_for))
        for arg in func.args_with_derivatives:
            if arg.type in ['TensorList', 'const c10::List<c10::optional<Tensor>> &']:
                setup.append(f'grad_fn->{arg.name}_size_ = {arg.name}.size();')

        return setup

    def setup_derivative(differentiable_inputs):
        env = {}
        env['args_with_derivatives'] = [arg['name'] for arg in args_with_derivatives]
        env['op'] = func.op if func is not None else 'NotImplemented'
        env['op_ctor'] = '' if func is not None else '"{}"'.format(declaration['api_name'])

        if is_out_fn:
            # For out functions, ensure that no input or output requires grad
            body = []
            body.append(DECLARE_GRAD_FN.substitute(op='Node'))
            body.append(SETUP_NONE_REQUIRES_GRAD.substitute(
                base_name=base_name,
                args_to_check=[arg['name'] for arg in differentiable_inputs]))
            body.append(SETUP_NONE_REQUIRES_GRAD.substitute(
                base_name=base_name,
                args_to_check=[arg['name'] for arg in differentiable_outputs]))
            return body

        setup = []
        setup.extend(ASSIGN_GRAD_FN.substitute(env).split('\n'))
        setup.extend(emit_save_inputs())

        body = []
        body.extend(emit_check_no_requires_grad(differentiable_inputs, args_with_derivatives))
        body.append(DECLARE_GRAD_FN.substitute(env))
        body.append(SETUP_DERIVATIVE.substitute(setup=setup))
        return body

    def emit_check_if_in_complex_autograd_allowlist():
        body = []
        if base_name in GRADIENT_IMPLEMENTED_FOR_COMPLEX:
            return body
        for arg in differentiable_outputs:
            name = arg['name']
            if arg['type'] in ['Tensor', 'TensorList', 'const c10::List<c10::optional<Tensor>> &']:
                body.append('throw_error_for_complex_autograd({}, "{}");'.format(name, base_name))
        return body

    def emit_check_no_requires_grad(tensor_args, args_with_derivatives):
        """Checks that arguments without derivatives don't require grad"""
        body = []
        for arg in tensor_args:
            if arg in args_with_derivatives:
                continue
            name = arg['name']
            if name in non_differentiable_arg_names:
                continue
            if name == 'output':
                # Double-backwards definitions sometimes take in 'input' and
                # 'output', but only define the derivative for input.
                continue
            if arg['dynamic_type'] in {'IndexTensor', 'ByteTensor', 'BoolTensor'}:
                continue
            body.append('check_no_requires_grad({}, "{}");'.format(name, name))
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

    def emit_dispatch_call(api_name, input_base, unpacked_args):
        """ Dispatch call via function in a namespace or method on Tensor."""
        if 'namespace' in declaration['method_of']:
            if declaration['use_c10_dispatcher'] in ['hacky_wrapper_for_legacy_signatures', 'full']:
                dispatcher_api_name = make_out_api_name_faithful(api_name)
            else:
                assert declaration['use_c10_dispatcher'] == 'with_codegenerated_unboxing_wrapper'
                dispatcher_api_name = api_name
            call = CALL_DISPATCH_VIA_NAMESPACE.substitute(
                api_name=dispatcher_api_name,
                unpacked_args=unpacked_args)
        else:
            call = CALL_DISPATCH_VIA_METHOD.substitute(
                api_name=api_name,
                var=input_base,
                unpacked_method_args=unpacked_args[1:])
        return call

    def emit_view_lambda():
        """ Generate an additional lambda function to recover views in backward when as_strided is not supported.
        See Note [View + Inplace update for base tensor] and [View + Inplace update for view tensor] for more details."""
        input_base = 'input_base'
        replay_view_func = ''
        updated_unpacked_args = []
        combined = nested_dict(env, declaration)
        known_view_arg_simple_types = ['int64_t', 'int64_t?', 'bool', 'IntArrayRef']
        for arg in combined['unpacked_args']:
            if arg == 'self_':
                updated_unpacked_args.append(input_base)
                continue
            arg_type = combined['unpacked_args_simple_type'][arg]
            if arg_type not in known_view_arg_simple_types:
                raise TypeError('You are adding an {} {} argument to op {} in addition to known types: {}. '
                                'Please update the list or materialize it so that it can be closed over by value, '
                                'also add a test in pytorch/xla/test/test_operations.py where this code is exercised.'
                                .format(arg_type, arg, declaration['name'], ', '.join(known_view_arg_simple_types)))

            if arg_type == 'IntArrayRef':
                # It's not safe to close over IntArrayRef by value, since this is a
                # reference type, so materialize a vector to close over by value
                arg_vec = arg + '_vec'
                replay_view_func += ARRAYREF_TO_VEC.substitute(arg=arg, vec=arg_vec)
                updated_unpacked_args.append(arg_vec)
            elif arg_type == 'int64_t?':
                # Materialize int64_t? to int64_t
                arg_value = arg + '_val'
                replay_view_func += OPTIONAL_TO_VAL.substitute(arg=arg, val=arg_value, default='0')
                updated_unpacked_args.append(arg_value)
            else:
                updated_unpacked_args.append(arg)

        replay_view_call = emit_dispatch_call(combined['api_name'], input_base, updated_unpacked_args)
        replay_view_func += REPLAY_VIEW_LAMBDA_FUNC.substitute(
            input_base=input_base,
            replay_view_call=replay_view_call)

        is_view_with_metadata_change = 'true' if name in VIEW_FUNCTIONS_WITH_METADATA_CHANGE else 'false'

        return SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE.substitute(
            is_view_with_metadata_change=is_view_with_metadata_change,
            replay_view_func=replay_view_func)

    def wrap_output(return_values, var):
        call = ''
        rhs_value = None
        if 'Tensor' not in declaration['return_type']:
            rhs_value = var
        elif view_info is not None:
            # See NOTE [ Autograd View Variables ] in variable.h for details.
            differentiable_output_vars = {r['name'] for r in differentiable_outputs}

            if not isinstance(view_info, str):
                raise TypeError("The view info should be a string for {}, but it is: {}".format(base_name, view_info))

            if len(differentiable_output_vars) == 0:
                # no output is differentiable (.indices() for SparseTensors for example)
                rhs_value = f'as_view({view_info}, {var}, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false)'
            elif len(differentiable_output_vars) == 1:
                # Single differentiable output (Tensor or Tensor[])
                return_info = differentiable_outputs[0]
                # We only support simple Tensor or a TensorList for functions that return views
                if not return_info['dynamic_type'] in ['Tensor', 'TensorList']:
                    raise RuntimeError("{} that return differentiable views can only return Tensor or Tensor[]".format(base_name))
                # Only allow rebasing of the history if we return a single Tensor
                # If we are in a no grad block, raise a warning
                # See NOTE [ View + Inplace detection ] for more details about this logic
                if return_info['dynamic_type'] in ['TensorList', 'const c10::List<c10::optional<Tensor>> &']:
                    if base_name in MULTI_OUTPUT_SAFE_FUNCTIONS:
                        creation_meta = "CreationMeta::MULTI_OUTPUT_SAFE"
                    else:
                        creation_meta = "CreationMeta::MULTI_OUTPUT_NODE"
                    call += ("as_view(/* base */ {}, /* output */ {}, /* is_bw_differentiable */ true, "
                             "/* is_fw_differentiable */ true, "
                             "/* creation_meta */ {});").format(view_info, var, creation_meta)
                    rhs_value = 'std::move({})'.format(var)
                else:
                    call += emit_view_lambda()
                    creation_meta = "GradMode::is_enabled() ? CreationMeta::DEFAULT: CreationMeta::NO_GRAD_MODE"
                    rhs_value = ("as_view(/* base */ {}, /* output */ {}, /* is_bw_differentiable */ true, "
                                 "/* is_fw_differentiable */ true, "
                                 "/* view_func */ func, /* creation_meta */ {})").format(view_info, var, creation_meta)
            else:
                # This could be supported but we don't need it at the moment, so keeping things simple.
                raise RuntimeError("Function that return multiple differentiable output "
                                   "when at least one of them is view is not supported.")
        else:
            rhs_value = 'std::move({})'.format(var)
        assert rhs_value is not None
        call += ASSIGN_RETURN_VALUE.substitute(return_values=return_values,
                                               rhs_value=rhs_value)
        return call

    def enforce_same_tensorimpl_and_storage(env, call):
        save_ptrs_stmts = []
        enforce_same_ptrs_stmts = []
        if declaration['name'] not in DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE:
            for arg in env.get('unpacked_args', []):
                simple_type = env['unpacked_args_simple_type'][arg]
                if simple_type == 'TensorList':
                    save_ptrs_stmts += [SAVE_TENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                                        SAVE_TENSORLIST_IMPL.substitute(tensorlist_name=arg)]
                    enforce_same_ptrs_stmts += [ENFORCE_SAME_TENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                                                ENFORCE_SAME_TENSORLIST_IMPL.substitute(tensorlist_name=arg)]
                elif simple_type == 'c10::List<c10::optional<Tensor>>':
                    save_ptrs_stmts += [SAVE_OPTIONALTENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                                        SAVE_OPTIONALTENSORLIST_IMPL.substitute(tensorlist_name=arg)]
                    enforce_same_ptrs_stmts += [ENFORCE_SAME_OPTIONALTENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                                                ENFORCE_SAME_OPTIONALTENSORLIST_IMPL.substitute(tensorlist_name=arg)]
                elif simple_type == 'Tensor':
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

    def emit_call(env, tie_return_values):
        combined = nested_dict(env, declaration)
        # We only care about adding `at::AutoNonVariableTypeMode` guard for non-variable dispatch
        # (which corresponds to 'use_derived' strategy). The purpose of this guard is to make sure
        # the baseType operations still dispatch to non-Variable type, even if the arguments passed
        # in are now Variables.
        # See NOTE [ Treating Variables as non-Variables in type dispatch ] for details.
        base_type_call = emit_dispatch_call(combined['api_name'], 'self_', combined['unpacked_args'])
        if not modifies_arguments and not returns_void:
            call = DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES.substitute(
                base_type_call=base_type_call)

            call += wrap_output(tie_return_values, 'tmp')
        else:
            call = DISPATCH_TO_NON_VAR_TYPE_WITHOUT_RETURN_VALUES.substitute(
                base_type_call=base_type_call)
        call = enforce_same_tensorimpl_and_storage(env, call)
        return call

    def emit_history():
        fn = 'rebase' if modifies_arguments and view_info is None else 'set'
        output_names = [r['name'] for r in differentiable_outputs]
        # TODO: flatten allocates a std::vector, which could be expensive
        outs = CodeTemplate("flatten_tensor_args( ${outs} )").substitute(outs=output_names)
        return SET_HISTORY.substitute(fn=fn, differentiable_outputs=outs)

    def emit_save_outputs():
        if is_out_fn:
            # out functions don't currently support differentiation
            return ''
        func = declaration['derivative']
        if func is not None:
            stmts = save_variables(func.all_saved_outputs, True)
            if len(stmts) == 0:
                return ''
            return CONDITIONAL.substitute(cond='grad_fn', statements=stmts)
        return ''

    def emit_any_requires_grad():
        return [SETUP_ANY_REQUIRES_GRAD.substitute(
            args_with_derivatives=[arg['name'] for arg in args_with_derivatives]), ]

    def emit_check_inplace():
        if not inplace:
            return []
        return ['check_inplace({}, _any_requires_grad);'.format(arg['name']) for arg in differentiable_outputs]

    def emit_increment_version():
        if not modifies_arguments:
            return []
        return ['increment_version({});'.format(arg['name']) for arg in returns]

    env = {}
    combined = nested_dict(env, declaration)

    body = []

    declare_returned_variables, tie_return_values, get_return_value = format_return_variables(declaration)

    body.extend(unpack_args(env, declaration))
    if requires_derivative:
        body.extend(emit_any_requires_grad())
        body.extend(emit_check_inplace())
        body.extend(setup_derivative(differentiable_inputs))
    body.append(declare_returned_variables)

    body.append(emit_call(env, tie_return_values))
    body.extend(emit_increment_version())
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
        body.append('return {};'.format(get_return_value))
    return body


def unpack_args(env, declaration):
    def requires_unpack(arg):
        return 'Tensor' in arg['dynamic_type'] and 'c10::optional' not in arg['type']

    body = []
    unpacked_args = []
    unpacked_args_simple_type = {}
    if declaration['use_c10_dispatcher'] in ['full', 'hacky_wrapper_for_legacy_signatures']:
        arguments = declaration['schema_order_arguments']
    else:
        assert declaration['use_c10_dispatcher'] == 'with_codegenerated_unboxing_wrapper'
        arguments = declaration['arguments']
    for i, arg in enumerate(arguments):
        if not requires_unpack(arg):
            unpacked_args.append(arg['name'])
            unpacked_args_simple_type[arg['name']] = arg['simple_type']
            continue

        dynamic_type = arg['dynamic_type']
        if 'TensorOptions' not in dynamic_type:
            is_nullable = arg.get('is_nullable', False)
            ref = (not is_nullable) and dynamic_type != 'TensorList'
            suffix = '_opt' if is_nullable and dynamic_type != 'TensorList' else ''
            body.append(UNPACK_TENSOR.substitute(
                arg_name=arg['name'],
                arg_pos=i,
                suffix=suffix,
                ref='&' if ref else '',
            ))
        else:
            # Okay, we are abusing the definition of 'unpack' here a bit,
            # although it's still getting the non-variable from the variable
            # (in this case via TensorOptions rather than Variable/Tensor).
            assert declaration['use_c10_dispatcher'] == 'with_codegenerated_unboxing_wrapper', \
                "VariableKernel shouldn't take TensorOptions if the op is c10-full"
            body.append(LEGACY_WRAP_OPTIONS.substitute(arg_name=arg['name']))

        unpacked_args.append(arg['name'] + '_')
        unpacked_args_simple_type[arg['name'] + '_'] = arg['simple_type']

    env['unpacked_args'] = unpacked_args
    env['unpacked_args_simple_type'] = unpacked_args_simple_type
    return body


def dispatch_strategy(declaration):
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
    if declaration['abstract'] or declaration['derivative'] is not None:
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

def get_decl_signature(declaration: Dict[Any, Any], use_base_variant: bool = False) -> str:
    name = declaration['name']
    arguments = declaration['arguments']
    if use_base_variant:
        if declaration['inplace']:
            assert name.endswith('_')
            name = name[:-1]
        elif name.endswith('_out'):
            name = name[:-4]
            arguments = [arg for arg in arguments if not arg.get('output', False)]
    simple_types = ', '.join(arg['simple_type'] for arg in arguments)
    return f'{name}({simple_types})'

@with_native_function
def get_func_signature(f: NativeFunction) -> str:
    args = CppSignatureGroup.from_native_function(f, method=False).signature.arguments()
    types = ', '.join(python.argument_type_str(a.argument.type, simple_type=True)
                      if isinstance(a.argument, Argument) else 'TensorOptions'
                      for a in args)
    return f'{cpp.name(f.func)}({types})'

def match_declarations_with_differentiability_info(
    declarations: Dict[Any, Any],
    differentiability_infos: Sequence[DifferentiabilityInfo],
) -> None:
    """Sets the "derivative" key on declarations to matching autograd function

    In-place functions will use the out-of-place derivative definition if there
    is no in-place specific derivative.
    """

    info_by_signature = {get_func_signature(info.func): info for info in differentiability_infos}

    def find_info(declaration: Dict[Any, Any]) -> Optional[DifferentiabilityInfo]:
        signature = get_decl_signature(declaration)
        if signature in info_by_signature:
            return info_by_signature[signature]

        # if there is no exact match look for the out-of-place signature.
        # i.e mul() for mul_() or mul_out()
        signature = get_decl_signature(declaration, use_base_variant=True)
        return info_by_signature.get(signature)

    for declaration in declarations:
        info = find_info(declaration)
        declaration['derivative'] = info if info and info.args_with_derivatives else None

        # Currently, the '.strides()' to 'strides_or_error' replacement does not support
        # 'self' derivatives of an inplace function, so we must check for this case.
        if declaration['inplace'] and (info is not None):
            for derivative in info.derivatives:
                if 'self' in derivative.var_names:
                    for saved_input in derivative.saved_inputs:
                        assert 'strides_or_error' not in saved_input.expr, (
                            "Calling '.strides()' in the 'self' derivative formula of an "
                            f"in-place function is not supported: {declaration['name']}")

        declaration['non_differentiable_arg_names'] = info.non_differentiable_arg_names if info else []
        declaration['output_differentiability'] = info.output_differentiability if info else None
