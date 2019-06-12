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
from .gen_autograd import VIEW_FUNCTIONS
from .gen_autograd_functions import uses_single_grad


# These functions are written manually in templates/VariableType.cpp
MANUAL_IMPLEMENTATIONS = {
    'resize_', 'resize_as_', 'detach', 'detach_',
}

# These functions we don't want to record for tracing, because we always want
# to trace their constituent parts.  This is a temporary hack in lieue
# of proper scopes, where subsequent compilation passes can ask for the unfolding
# on demand.  Only concrete ATen methods can be disabled this way; it will have
# NO EFFECT otherwise.
DONT_RECORD_TRACE = {
    'convolution', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
    'conv_transpose2d', 'conv_transpose3d', 'lstm_cell', 'gru_cell',
    'rnn_tanh_cell', 'rnn_relu_cell', 'linear',
    # FIXME: figure out a better way when we support sparse tensors in jit
    '_coalesced_',
}

# These functions have their names recorded under trace renamed,
RENAME_TRACE = {
    'zero': 'zeros_like',
    'fill': 'full_like',
}

# (declaration name, argument name) -> attribute name
RENAME_ATTRIBUTES = {
    ('fill_', 'value'): 'fill_value'
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
    # These only depend on the input Tensor's shape and device, not the data
    'ones_like', 'zeros_like', 'rand_like', 'randn_like',
    # These are only implemented on integral types
    '__and__', '__iand__', '__ilshift__', '__ior__', '__irshift__', '__ixor__',
    '__lshift__', '__or__', '__rshift__', '__xor__',
    # This is an unsafe method that is meant to be out of reach of autograd.
    '_coalesced_',
}

METHOD_DECLARATION = CodeTemplate("""\
${return_type} ${method_prefix_derived}${api_name}(${type_method_formals}) const override;
""")

METHOD_DEFINITION = CodeTemplate("""\
${return_type} VariableType::${method_prefix_derived}${api_name}(${type_method_formals}) const {
  ${type_definition_body}
}
""")

UNPACK_TENSOR = CodeTemplate("""\
auto${ref} ${arg_name}_ = unpack${suffix}(${arg_name}, "${arg_name}", ${arg_pos});""")

UNPACK_OPTIONS = CodeTemplate("""\
auto ${arg_name}_ = TensorOptions(${arg_name}).is_variable(false);""")

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
TypeDefault::${method_prefix_derived}${api_name}(${type_method_args})""")

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
profiler::RecordFunction profiler("${name}", Function::peek_at_next_sequence_nr());""")

PRE_RECORD_TRACE = CodeTemplate("""\
torch::jit::Node* node = nullptr;
std::shared_ptr<jit::tracer::TracingState> tracer_state;
if (jit::tracer::isTracing()) {
  tracer_state = jit::tracer::getTracingState();
  const static auto op_name = jit::Symbol::fromQualString("aten::${trace_name}");
  node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
  jit::tracer::recordSourceLocation(node);
  ${add_trace_inputs}
  tracer_state->graph->appendNode(node);
  ${inplace_guard}
  jit::tracer::setTracingState(nullptr);
}
""")

INPLACE_GUARD = CodeTemplate("""\
jit::tracer::ensureUnique("${name}", ${mutable_input});
""")

ADD_TRACE_INPUT = CodeTemplate("""jit::tracer::addInputs(node, "${name}", ${input});""")

POST_RECORD_TRACE = CodeTemplate("""\
if (tracer_state) {
  jit::tracer::setTracingState(std::move(tracer_state));
  ${record_trace_outputs}
}
""")


FACTORY_FUNCTION_NAMES = None


def find_factory_functions(declarations):
    global FACTORY_FUNCTION_NAMES
    FACTORY_FUNCTION_NAMES = set()

    for declaration in declarations:
        if declaration['is_factory_method']:
            FACTORY_FUNCTION_NAMES.add(declaration['api_name'])


def should_trace(declaration):
    # Operations involving Storage or Type are not traceable at the moment
    if any(arg['simple_type'] in {'Storage', 'Type'} for arg in declaration['arguments']):
        return False
    # We can't trace functions which don't have any Tensor or TensorList returns
    if 'Tensor' not in declaration['return_type']:
        return False
    name = declaration['name']
    base_name = name[:-1] if declaration['inplace'] else name[:-4] if name.endswith('_out') else name
    if base_name in DONT_RECORD_TRACE or name in DONT_RECORD_TRACE:
        return False
    return True


def record_trace_outputs(declaration):
    if declaration['name'].endswith('_out'):
        output_names = [arg['name'] for arg in declaration['arguments'] if arg.get('output', False)]
    else:
        output_names = [r['name'] for r in declaration['returns']]
    return ['jit::tracer::addOutput(node, {});'.format(n) for n in output_names]


def format_trace(declaration):
    local = {}
    local['trace_name'] = trace_name = uninplace_api_name(declaration['api_name'])

    # *_out functions take the result as a first argument, but since we're
    # going to de-inplace the call, we need to remove it from the argument list
    trace_inputs = declaration['arguments']
    if declaration['name'].endswith('_out'):
        trace_inputs = trace_inputs[1:]
    trace_input_spec = [(i['name'], i['name']) for i in trace_inputs]

    # factories are a bit special because their out-of-place overloads
    # take an extra TensorOptions argument, which is missing in the _out function
    has_factory_name = trace_name in FACTORY_FUNCTION_NAMES
    is_out_overload = any(arg['name'] == 'result' for arg in declaration['arguments'])
    if has_factory_name and is_out_overload:
        trace_input_spec.append(('result', 'result.options()'))

    local['add_trace_inputs'] = \
        '\n'.join(ADD_TRACE_INPUT.substitute(name=name, input=value) for name, value in trace_input_spec)

    # Record inplace operations as out-of-place operations (e.g.,
    # not add_ but add)
    # TODO: Add a proper concept of side effects to the IR, and
    # properly record inplace operations.
    local['inplace_guard'] = ''
    if local['trace_name'] != declaration['api_name']:
        local['inplace_guard'] = INPLACE_GUARD.substitute(name=declaration['api_name'],
                                                          mutable_input=declaration['arguments'][0]['name'])
    if local['trace_name'] in RENAME_TRACE:
        local['trace_name'] = RENAME_TRACE[local['trace_name']]

    local['record_trace_outputs'] = record_trace_outputs(declaration)

    return (PRE_RECORD_TRACE.substitute(local), POST_RECORD_TRACE.substitute(local))


def gen_variable_type(out, aten_declarations, template_path):
    """VariableType.h and VariableType.cpp body

    This is the at::Type subclass for differentiable tensors. The
    implementation of each function dispatches to the base tensor type to
    compute the output. The grad_fn is attached to differentiable functions.
    """

    # WARNING: this function call modifies global mutable state
    find_factory_functions(aten_declarations)

    aten_declarations = list(sorted(aten_declarations, key=lambda decl: decl['name']))

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

    for declaration in aten_declarations:
        # Factory methods usually do not appear in `VariableType` at all, since they
        # don't dispatch via `Type`; except in the case where the implementation is 'abstract'
        # in which case they do!
        if declaration['is_factory_method']:
            continue
        type_declarations.append(METHOD_DECLARATION.substitute(declaration))
        if declaration['name'] not in MANUAL_IMPLEMENTATIONS:
            type_definitions.append(emit_method_definition(declaration))

    env = {
        'type_derived_method_declarations': type_declarations,
        'type_derived_method_definitions': type_definitions,
    }
    if header:
        write(out, 'VariableType.h', VARIABLE_TYPE_H, env)
    else:
        write(out, 'VariableType%s.cpp' % suffix, VARIABLE_TYPE_CPP, env)


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
    returns_void = len(returns) == 1 and returns[0]['type'] == 'void'

    base_name = name[:-1] if inplace else name[:-4] if is_out_fn else name
    view_info = VIEW_FUNCTIONS.get(base_name, None)

    # These exclude things like BoolTensor, int64_t, and Scalar
    def is_differentiable(arg):
        if 'TensorOptions' in arg['type']:
            return False
        if 'Tensor' not in arg['type']:
            return False
        if arg['dynamic_type'] in {'IndexTensor', 'BoolTensor'}:
            return False
        return True

    inputs = [arg for arg in arguments if not arg.get('output', False)]
    differentiable_inputs = list(filter(is_differentiable, inputs))
    candidate_differentiable_outputs = list(filter(is_differentiable, returns))

    if func is not None and func.get('output_differentiability') is not None:
        differentiable_outputs = []
        output_differentiability = func.get('output_differentiability')
        for differentiable, output in zip(output_differentiability, returns):
            if differentiable:
                differentiable_outputs.append(output)
    elif uses_single_grad(func):
        differentiable_outputs = candidate_differentiable_outputs[:1]
    else:
        differentiable_outputs = candidate_differentiable_outputs

    requires_derivative = (
        base_name not in DONT_REQUIRE_DERIVATIVE and name not in DONT_REQUIRE_DERIVATIVE and
        len(differentiable_inputs) > 0 and len(differentiable_outputs) > 0 and
        strategy == 'use_derived')

    if func is not None and not requires_derivative:
        print('WARNING: derivative ignored for {}'.format(name), file=sys.stderr)

    def setup_derivative():
        args_with_derivatives = find_args_with_derivatives()

        env = {}
        env['args_with_derivatives'] = reference_args(args_with_derivatives)
        env['op'] = func['op'] if func is not None else 'NotImplemented'
        env['op_ctor'] = '' if func is not None else '"{}"'.format(declaration['api_name'])

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
            elif arg['type'] == 'IntList':
                expr = expr + ".vec()"
            stmts.append('grad_fn->{} = {};'.format(name, expr))
        return stmts

    def reference_args(args):
        res = []
        for arg in args:
            if arg['type'] == 'SparseTensorRef':
                res.append('{}.tref'.format(arg['name']))
            else:
                res.append(arg['name'])
        return res

    def emit_record_trace(env):
        if not should_trace(declaration):
            return ('', '')
        return format_trace(declaration)

    def declare_returned_variables():
        if modifies_arguments:
            return ''
        if len(declaration['returns']) == 1:
            return ''
        # TODO: this will be ugly
        names = [ret['type'] + ' ' + ret['name'] + ';' for ret in declaration['returns']]
        return '\n'.join(names)

    def wrap_output(call):
        # Returns a 2-tuple `(wrapped_call, extra_wrapping_stmts)`, where
        # `wrapped_call` is to drop-in replace `call`, and
        # `extra_wrapping_stmts` is a list of extra statements to run after
        # `call`.
        if 'Tensor' not in declaration['return_type']:
            return call, []
        elif view_info is not None:
            # See NOTE [ Autograd View Variables ] in variable.h for details.
            differentiable_output_vars = {r['name'] for r in differentiable_outputs}
            tensor_output_vars = {r['name'] for r in returns if 'Tensor' in r['type']}
            if not isinstance(view_info, dict):
                if len(differentiable_output_vars) == len(tensor_output_vars):
                    # all outputs are differentiable
                    return 'as_view({}, {}, true)'.format(view_info, call), []
                elif len(differentiable_output_vars) == 0:
                    # no output is differentiable
                    return 'as_view({}, {}, false)'.format(view_info, call), []
                else:
                    # some of the outputs are differentiable
                    # need to expand to dict mode, i.e., one entry per output
                    base_name = view_info
                    view_info_dict = {}
                    for i, return_info in enumerate(returns):
                        if 'Tensor' in return_info['type']:
                            view_info_dict[i] = base_name
            else:
                view_info_dict = view_info

            def wrap_view_single(output_var, base_var):
                fmt = '{output_var} = as_view({base_var}, {output_var}, {is_differentiable});'
                if output_var in differentiable_output_vars:
                    # If `GradMode::is_enabled()` is False, this is a
                    # non-differentiable view. Gradients should not flow through.
                    is_differentiable = 'true'
                else:
                    # This output is non-differentiable, so it is a
                    # non-differentiable view. Gradients should not flow through.
                    is_differentiable = 'false'
                return fmt.format(output_var=output_var, base_var=base_var,
                                  is_differentiable=is_differentiable)

            extra_wrapping_stmts = []
            for output_idx, return_info in enumerate(returns):
                if 'Tensor' not in return_info['type']:
                    assert output_idx not in view_info_dict, 'Can not wrap non-Tensor output as a view'
                    continue
                output_var = return_info['name']
                if output_idx in view_info_dict:
                    stmt = wrap_view_single(output_var, view_info_dict[output_idx])
                elif 'Tensor' in return_info['type']:
                    stmt = '{output_var} = as_variable({output_var});'.format(output_var=output_var)
                extra_wrapping_stmts.append(stmt)
            return call, extra_wrapping_stmts
        else:
            return 'as_variable({})'.format(call), []

    def emit_call(env):
        combined = nested_dict(env, declaration)
        extra_wrapping_stmts = []
        if strategy == 'use_derived':
            call = CALL_VIA_DERIVED.substitute(combined)
            if not modifies_arguments:
                call, extra_wrapping_stmts = wrap_output(call)
        else:
            call = CALL_VIA_TYPE.substitute(declaration)
        if not modifies_arguments and not returns_void:
            call = '{} = {}'.format(tie_return_values(), call)
        call = call + ';'
        for stmt in extra_wrapping_stmts:
            call += '\n' + stmt
        return call

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
    if not returns_void:
        body.append('return {};'.format(get_return_value()))
    return body


def unpack_args(env, declaration):
    def requires_unpack(arg):
        return 'Tensor' in arg['dynamic_type']

    body = []
    unpacked_args = []
    for i, arg in enumerate(declaration['arguments']):
        if not requires_unpack(arg):
            unpacked_args.append(arg['name'])
            continue

        dynamic_type = arg['dynamic_type']
        if 'TensorOptions' not in dynamic_type:
            is_nullable = arg.get('is_nullable', False)
            ref = (not is_nullable) and dynamic_type not in ['TensorList', 'SparseTensorRef']
            suffix = '_opt' if is_nullable else ''

            body.append(UNPACK_TENSOR.substitute(
                arg_name=arg['name'],
                arg_pos=i,
                suffix=suffix,
                ref='&' if ref else '',
            ))
        else:
            # Okay, we are abusing the definition of 'unpack' here a bit,
            # although it's stll getting the non-variable from the variable
            # (in this case via TensorOptions rather than Variable/Tensor).
            body.append(UNPACK_OPTIONS.substitute(arg_name=arg['name']))

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
    if (declaration['abstract'] or declaration['requires_tensor'] or
            declaration['derivative'] is not None):
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
