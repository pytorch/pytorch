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
from __future__ import print_function
import os
import sys
from .utils import CodeTemplate, nested_dict, write, uninplace_api_name
from .gen_autograd import VIEW_FUNCTIONS, template_path, \
    HARDCODED_DIFFERENTIABLE_OUTPUTS
from .gen_autograd_functions import uses_single_grad

VARIABLE_TYPE_H = CodeTemplate.from_file(template_path + '/VariableType.h')
VARIABLE_TYPE_CPP = CodeTemplate.from_file(template_path + '/VariableType.cpp')

# These functions are written manually in templates/VariableType.cpp
MANUAL_IMPLEMENTATIONS = {
    'contiguous', 'resize_', 'resize_as_'
}

# These functions we don't want to record for tracing, because we always want
# to trace their constituent parts.  This is a temporary hack in lieue
# of proper scopes, where subsequent compilation passes can ask for the unfolding
# on demand.  Only concrete ATen methods can be disabled this way; it will have
# NO EFFECT otherwise.
DONT_RECORD_TRACE = {
    'convolution', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
    'conv_transpose2d', 'conv_transpose3d',
}

# These functions are not worth profiling because they are very cheap and may
# be called very often.
DONT_PROFILE = {
    'data_ptr', 'get_device', 'is_contiguous', 'is_cuda', 'is_distributed',
    'is_same_size', 'is_set_to', 'is_signed', 'is_sparse', 'numel',
    'size', 'storage_offset', 'stride',
}

# We don't set or modify grad_fn on these methods. Generally, they return
# tensors that have requires_grad=False. In-place functions listed here will
# not examine or modify requires_grad or grad_fn.
DONT_REQUIRE_DERIVATIVE = {
    # These  only depend on the input Tensor's shape and device, not the data
    'ones_like', 'zeros_like', 'rand_like', 'randn_like',
    # Tensor constructors
    'sparse_coo_tensor',
    # These are only implemented on integral types
    '__and__', '__iand__', '__ilshift__', '__ior__', '__irshift__', '__ixor__',
    '__lshift__', '__or__', '__rshift__', '__xor__',
}

METHOD_DECLARATION = CodeTemplate("""\
virtual ${return_type} ${method_prefix_derived}${api_name}(${type_method_formals}) const override;
""")

METHOD_DEFINITION = CodeTemplate("""\
${return_type} VariableType::${method_prefix_derived}${api_name}(${type_method_formals}) const {
  ${type_definition_body}
}
""")

UNPACK_TENSOR = CodeTemplate("""\
auto${ref} ${arg_name}_ = unpack${suffix}(${arg_name}, "${arg_name}", ${arg_pos});""")

DECLARE_GRAD_FN = CodeTemplate("""\
std::shared_ptr<${op}> grad_fn;
""")

SETUP_DERIVATIVE = CodeTemplate("""\
if (compute_requires_grad( ${args_with_derivatives} )) {
  ${setup}
}
""")

ASSIGN_GRAD_FN = CodeTemplate("""\
grad_fn = std::shared_ptr<${op}>(new ${op}(${op_ctor}), deleteFunction);
grad_fn->set_next_edges(collect_next_edges( ${args_with_derivatives} ));
""")

CALL_VIA_TYPE = CodeTemplate("""\
Type::${method_prefix_derived}${api_name}(${type_method_args})""")

CALL_VIA_DERIVED = CodeTemplate("""\
baseType->${method_prefix_derived}${base_name}(${unpacked_args})""")

SET_HISTORY = CodeTemplate("""\
${fn}_history(${differentiable_outputs}, grad_fn);
""")

CONDITIONAL = CodeTemplate("""\
if (${cond}) {
  ${statements}
}
""")

RECORD_FUNCTION = CodeTemplate("""\
profiler::RecordFunction profiler("${name}");""")

PRE_RECORD_TRACE = CodeTemplate("""\
jit::tracer::PreTraceInfo trace_info;
if (jit::tracer::isTracing( ${tensor_args} )) {
  trace_info = jit::tracer::preRecordTrace( jit::aten::${trace_name}, ${trace_inputs} );
  if (!jit::tracer::ArgumentStash::empty()) {
    ${record_positional_attributes}
    TORCH_ASSERT(jit::tracer::ArgumentStash::empty());
  } else {
    ${record_attributes}
  }
}
""")

POST_RECORD_TRACE = CodeTemplate("""\
if (trace_info.state != nullptr) {
  jit::tracer::postRecordTrace( trace_info,  ${trace_outputs} );
}
""")

RECORD_ATTRIBUTE = CodeTemplate("""\
setattr(trace_info.n, jit::attr::${name}, ${name});""")

RECORD_POSITIONAL_ATTRIBUTE = CodeTemplate("""\
setposattr(trace_info.n, ${i}, "${name}", ${name});""")

POSITIONAL_ATTR_NYI = """\
throw std::runtime_error("Can't have size-dependent arguments to functions that "
                         "take variable number of tensor arguments");
"""


def should_trace(declaration):
    # Operations involving Generator, Storage, Type are not traceable
    # at the moment
    if any(arg['simple_type'] in {'Generator', 'Storage', 'ScalarType', 'Type', 'optional<ScalarType>'}
            for arg in declaration['arguments']):
        return False
    # We can't trace functions which don't have any Tensor or TensorList returns
    if 'Tensor' not in declaration['return_type']:
        return False
    tensor_args = [arg for arg in declaration['arguments'] if arg['simple_type'] in {'Tensor', 'TensorList'}]
    if len(tensor_args) == 0:
        return False
    name = declaration['name']
    base_name = name[:-1] if declaration['inplace'] else name[:-4] if name.endswith('_out') else name
    if base_name in DONT_RECORD_TRACE:
        return False
    return True


def gen_variable_type(out, aten_declarations):
    """VariableType.h and VariableType.cpp body

    This is the at::Type subclass for differentiable tensors. The
    implementation of each function dispatches to the base tensor type to
    compute the output. The grad_fn is attached to differentiable functions.
    """

    type_declarations = []
    type_definitions = []

    for declaration in aten_declarations:
        type_declarations.append(METHOD_DECLARATION.substitute(declaration))
        if declaration['name'] not in MANUAL_IMPLEMENTATIONS:
            type_definitions.append(emit_method_definition(declaration))

    env = {
        'type_derived_method_declarations': type_declarations,
        'type_derived_method_definitions': type_definitions,
    }
    write(out, 'VariableType.h', VARIABLE_TYPE_H, env)
    write(out, 'VariableType.cpp', VARIABLE_TYPE_CPP, env)


def emit_method_definition(declaration):
    body = emit_body(declaration)
    return METHOD_DEFINITION.substitute(declaration, type_definition_body=body)


def emit_body(declaration):
    strategy = dispatch_strategy(declaration)

    arguments = declaration['arguments']
    returns = declaration['returns']
    func = declaration['derivative']
    name = declaration['name']
    inplace = declaration['inplace']
    is_out_fn = name.endswith('_out')
    modifies_arguments = inplace or is_out_fn

    base_name = name[:-1] if inplace else name[:-4] if is_out_fn else name
    is_view = base_name in VIEW_FUNCTIONS

    # These exclude things like BoolTensor, int64_t, and Scalar
    def is_differentiable(arg):
        if 'Tensor' not in arg['type']:
            return False
        if arg['dynamic_type'] in {'IndexTensor', 'BoolTensor'}:
            return False
        return True

    inputs = [arg for arg in arguments if not arg.get('output', False)]
    differentiable_inputs = list(filter(is_differentiable, inputs))
    candidate_differentiable_outputs = list(filter(is_differentiable, returns))

    hardcoded_diff = HARDCODED_DIFFERENTIABLE_OUTPUTS.get(name)
    if hardcoded_diff:
        differentiable_outputs = []
        for i in hardcoded_diff:
            differentiable_outputs.append(candidate_differentiable_outputs[i])
    elif uses_single_grad(func):
        differentiable_outputs = candidate_differentiable_outputs[:1]
    else:
        differentiable_outputs = candidate_differentiable_outputs

    requires_derivative = (
        base_name not in DONT_REQUIRE_DERIVATIVE and
        len(differentiable_inputs) > 0 and len(differentiable_outputs) > 0 and
        strategy == 'use_derived')

    if func is not None and not requires_derivative:
        print('WARNING: derivative ignored for {}'.format(name), file=sys.stderr)

    def setup_derivative():
        def error_msg():
            name = declaration['api_name']
            return '"the derivative for {} is not implemented"'.format(name)

        args_with_derivatives = find_args_with_derivatives()

        env = {}
        env['args_with_derivatives'] = reference_args(args_with_derivatives)
        env['op'] = func['op'] if func is not None else 'Error'
        env['op_ctor'] = '' if func is not None else error_msg()

        if is_out_fn:
            setup = ['throw_error_out_requires_grad("{}");'.format(base_name)]
            body = []
            body.append(DECLARE_GRAD_FN.substitute(op='Function'))
            body.append(SETUP_DERIVATIVE.substitute(
                setup=setup,
                args_with_derivatives=reference_args(differentiable_inputs)))
            body.append(SETUP_DERIVATIVE.substitute(
                setup=setup,
                args_with_derivatives=reference_args(differentiable_outputs)))
            return body

        setup = []
        setup.extend(ASSIGN_GRAD_FN.substitute(env).split('\n'))
        if func is not None:
            setup.extend(save_variables(func['saved_inputs'], False))
            for arg in func['args_with_gradients']:
                if arg['type'] == 'TensorList':
                    setup.append("grad_fn->{}_size_ = {}.size();".format(arg['name'], arg['name']))

        body = []
        body.extend(emit_check_no_requires_grad(differentiable_inputs, args_with_derivatives))
        body.append(DECLARE_GRAD_FN.substitute(env))
        body.append(SETUP_DERIVATIVE.substitute(env, setup=setup))
        return body

    def find_args_with_derivatives():
        """Find arguments that have derivative definitions"""
        if func is None:
            return differentiable_inputs
        names = set(name for d in func['derivatives'] for name in d['var_names'])
        differentiable = [arg for arg in differentiable_inputs if arg['name'] in names]
        if len(differentiable) != len(names):
            missing = names - set(arg['name'] for arg in differentiable)
            raise RuntimeError('Missing arguments for derivatives: {} in {}'.format(missing, func['name']))
        return differentiable

    def emit_check_no_requires_grad(tensor_args, args_with_derivatives):
        """Checks that arguments without derivatives don't require grad"""
        body = []
        for arg in tensor_args:
            if arg in args_with_derivatives:
                continue
            name = arg['name']
            if name == 'output':
                # Double-backwards definitions sometimes take in 'input' and
                # 'output', but only define the derivative for input.
                continue
            if arg['dynamic_type'] in {'IndexTensor', 'BoolTensor'}:
                continue
            body.append('check_no_requires_grad({}, "{}");'.format(name, name))
        return body

    def save_variables(saved_variables, is_output):
        # assign the saved variables to the generated grad_fn
        stmts = []
        for arg in saved_variables:
            name = arg['name']
            expr = arg.get('expr', arg['name'])
            if arg['type'] == 'Tensor' or (is_output and arg['type'] == 'Scalar'):
                name += '_'
                var = arg['name']
                if var == 'self' and inplace:
                    var = 'self.clone()'
                    assert not is_output
                if inplace and is_output:
                    var = 'self'
                expr = 'SavedVariable({}, {})'.format(var, str(is_output).lower())
            elif arg['type'] == 'TensorList':
                name += '_'
                expr = 'make_saved_variable_list({})'.format(arg['name'])
            stmts.append('grad_fn->{} = {};'.format(name, expr))
        return stmts

    def reference_args(args):
        res = []
        for arg in args:
            if arg['type'] == 'SparseTensor':
                res.append('{}.tref'.format(arg['name']))
            else:
                res.append(arg['name'])
        return res

    def get_trace_outputs(declaration):
        if declaration['return_type'] == 'std::vector<Tensor>':
            return 'flatten({})'.format(declaration['returns'][0]['name'])
        elif name.endswith('_out'):
            output_args = [arg['name'] for arg in arguments
                           if arg.get('output', False)]
            return '{' + ', '.join(output_args) + '}'
        trace_outs = [r['name'] for r in declaration['returns']]
        if any(ret['dynamic_type'] == 'TensorList' for ret in declaration['returns']):
            return CodeTemplate("flatten( ${outs} )").substitute(outs=trace_outs)
        else:
            return CodeTemplate("{ ${outs} }").substitute(outs=trace_outs)

    def emit_record_trace(env):
        if not should_trace(declaration):
            return ('', '')

        # Note [clang-802.0.42 tuple overload bug]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Originally, my plan for emit_$ecord_trace was to keep it as
        # simple as possible, if at the expense of some somewhat ugly
        # overloads.  So this meant we had a 'recordTrace' function
        # with overloads like this:
        #
        #   recordTrace(..., const Variable& out)
        #   recordTrace(..., const std::tuple<Variable, Variable>& out)
        #
        # Unfortunately, this triggers a bug in clang-802.0.42
        # (widely used in macOS Sierra 10.12.6) wherein a Variable is
        # implicitly convertible into a std::tuple<Variable, Variable>;
        # a minimal repro can be seen below here:
        #
        #   #include <tuple>
        #   struct T {};
        #   void f(const std::tuple<T, T>&) {}
        #   void g(T& x) { f(x); }
        #
        # To work around this bug, the code generator is a bit more
        # complicated, and is taught how to handle this situation.

        local = {}

        tensor_args = [arg for arg in declaration['arguments'] if arg['simple_type'] in {'Tensor', 'TensorList'}]
        local['tensor_args'] = [arg['name'] for arg in tensor_args]
        if any(arg['simple_type'] == 'TensorList' for arg in tensor_args):
            # Allocate a temporary vector with flatten and pass it in
            local['trace_inputs'] = CodeTemplate("flatten( $tensor_args )").substitute(local)
        else:
            local['trace_inputs'] = CodeTemplate("{ ${tensor_args} }").substitute(local)

        local['record_attributes'] = []
        for arg in declaration['arguments']:
            if arg['simple_type'] in {'Tensor', 'TensorList'}:
                continue
            local['record_attributes'].append(RECORD_ATTRIBUTE.substitute(name=arg['name']))

        local['record_positional_attributes'] = []
        for i, arg in enumerate(declaration['arguments']):
            if arg['simple_type'] == 'Tensor':
                continue
            if arg['simple_type'] == 'TensorList':
                local['record_positional_attributes'] = POSITIONAL_ATTR_NYI
                break
            local['record_positional_attributes'].append(
                RECORD_POSITIONAL_ATTRIBUTE.substitute(name=arg['name'], i=i))

        # Record inplace operations as out-of-place operations (e.g.,
        # not add_ but add)
        # TODO: Add a proper concept of side effects to the IR, and
        # properly record inplace operations.
        local['trace_name'] = uninplace_api_name(declaration['api_name'])

        local['trace_outputs'] = get_trace_outputs(declaration)

        combined = nested_dict(local, nested_dict(env, declaration))
        return (PRE_RECORD_TRACE.substitute(combined), POST_RECORD_TRACE.substitute(combined))

    def declare_returned_variables():
        if modifies_arguments:
            return ''
        if len(declaration['returns']) == 1:
            return ''
        # TODO: this will be ugly
        names = [ret['type'] + ' ' + ret['name'] + ';' for ret in declaration['returns']]
        return '\n'.join(names)

    def wrap_output(call):
        if 'Tensor' not in declaration['return_type']:
            return call
        elif is_view:
            return 'as_view(self, {})'.format(call)
        else:
            return 'as_variable({})'.format(call)

    def emit_call(env):
        combined = nested_dict(env, declaration)
        if strategy == 'use_derived':
            call = CALL_VIA_DERIVED.substitute(combined)
            if not modifies_arguments:
                call = wrap_output(call)
        else:
            call = CALL_VIA_TYPE.substitute(declaration)
        if not modifies_arguments:
            call = '{} = {}'.format(tie_return_values(), call)
        return call + ';'

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

    def emit_history():
        fn = 'rebase' if modifies_arguments and not is_view else 'set'
        output_names = [r['name'] for r in differentiable_outputs]
        if len(output_names) == 1:
            outs = output_names[0]
        else:
            outs = CodeTemplate("{ ${outs} }").substitute(outs=output_names)
        return SET_HISTORY.substitute(fn=fn, differentiable_outputs=outs)

    def emit_save_outputs():
        if is_out_fn:
            # out functions don't currently support differentiation
            return ''
        func = declaration['derivative']
        if func is not None:
            stmts = save_variables(func['saved_outputs'], True)
            if len(stmts) == 0:
                return ''
            return CONDITIONAL.substitute(cond='grad_fn', statements=stmts)
        return ''

    def emit_check_inplace():
        if not inplace:
            return []
        return ['check_inplace({});'.format(arg['name']) for arg in differentiable_outputs]

    def emit_increment_version():
        if not modifies_arguments:
            return []
        return ['increment_version({});'.format(arg['name']) for arg in differentiable_outputs]

    env = {}
    combined = nested_dict(env, declaration)

    body = []
    if base_name not in DONT_PROFILE:
        body.append(RECORD_FUNCTION.substitute(combined))
    if strategy != 'use_type':
        body.extend(unpack_args(env, declaration))
    if requires_derivative:
        body.extend(emit_check_inplace())
        body.extend(setup_derivative())
    body.append(declare_returned_variables())

    pre_record_trace, post_record_trace = emit_record_trace(env)

    body.append(pre_record_trace)
    body.append(emit_call(env))
    if requires_derivative:
        # set_flags has to appear after version_counter, because rebase_history
        # requires that the counter is incremented before it is called
        body.extend(emit_increment_version())
        body.append(emit_history())
    # post_record_trace must appear before save_outputs so that saved outputs
    # have their tracing state saved (that is setup by recordTrace)
    body.append(post_record_trace)
    if requires_derivative:
        body.append(emit_save_outputs())
    body.append('return {};'.format(get_return_value()))
    return body


def unpack_args(env, declaration):
    def requires_unpack(arg):
        return 'Tensor' in arg['dynamic_type']

    body = []
    unpacked_args = []
    for i, arg in enumerate(declaration['arguments']):
        # these arguments are skipped from the Type method.
        if arg.get('is_type_dispatched'):
            continue
        if not requires_unpack(arg):
            unpacked_args.append(arg['name'])
            continue

        dynamic_type = arg['dynamic_type']
        is_nullable = arg.get('is_nullable', False)
        ref = (not is_nullable) and dynamic_type not in ['TensorList', 'SparseTensor']
        suffix = '_opt' if is_nullable else ''

        body.append(UNPACK_TENSOR.substitute(
            arg_name=arg['name'],
            arg_pos=i,
            suffix=suffix,
            ref='&' if ref else '',
        ))
        unpacked_args.append(arg['name'] + '_')

    env['unpacked_args'] = unpacked_args
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
    if (declaration['abstract'] or declaration['derivative'] is not None or
            any(arg.get('is_type_dispatched') for arg in declaration['arguments'])):
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
        # actually implemented  out of differentiable functions. (This
        # assumption might not hold, but then you'll see gradcheck fail.)
        return 'use_type'
