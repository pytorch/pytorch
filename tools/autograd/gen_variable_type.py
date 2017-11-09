import argparse
import copy
import os
import re
import yaml
from collections import defaultdict
from tools.shared.module_loader import import_module
from .nested_dict import nested_dict

CodeTemplate = import_module('code_template', 'aten/src/ATen/code_template.py').CodeTemplate


try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


METHOD_DECLARATION = CodeTemplate("""\
virtual ${return_type} ${method_prefix}${api_name}(${formals}) const override;
""")

METHOD_DEFINITION = CodeTemplate("""\
${return_type} VariableType::${method_prefix}${api_name}(${formals}) const {
    ${type_definition_body}
}
""")

METHOD_DEFINITION_NYI = CodeTemplate("""\
throw std::runtime_error("VariableType::${api_name} NYI");""")

BASE_CALL = CodeTemplate("""\
baseType->${method_prefix}${base_name}(${unpacked_args})""")

METHOD_DEFINITION_FALLTHROUGH = CodeTemplate("""\
${unpack_args}
return baseType->${method_prefix}${api_name}(${unpacked_args});""")

METHOD_DEFINITION_FALLTHROUGH_VARIABLE = CodeTemplate("""\
${unpack_args}
return as_variable(baseType->${method_prefix}${api_name}(${unpacked_args}));""")

METHOD_DEFINITION_FALLTHROUGH_INPLACE = CodeTemplate("""\
${unpack_args}
baseType->${method_prefix}${api_name}(${unpacked_args});
increment_version(self);
return self;
""")

UNPACK_TENSOR = CodeTemplate("""\
auto${ref} ${arg_name}_ = unpack${suffix}(${arg_name}, "${arg_name}", ${arg_pos});""")

FUNCTION_DECLARATION = CodeTemplate("""\
struct ${op} : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(const variable_list& grads) override;
  std::string name() override { return "${op}"; }
  void releaseVariables() override {
    ${release_variables}
  }
  ${saved_variables}
};
""")

FUNCTION_DEFINITION = CodeTemplate("""\
variable_list ${op}::apply(const variable_list& grads) {
  variable_list grad_inputs{${num_inputs}};
  ${body}
  return grad_inputs;
}
""")

PY_FUNCTION_DEFINITION = CodeTemplate("""\
static PyTypeObject ${op}Class;
addClass<${op}>(${op}Class, "${op}");
""")

DERIVATIVE_TENSOR = CodeTemplate("""\
if (should_compute_output(${idx})) {
  grad_inputs[${idx}] = ${derivative};
}
""")

DERIVATIVE_MULTI = CodeTemplate("""\
if (should_compute_output({ ${idxs} })) {
  auto output_mask = std::array<bool, ${n}>{
    ${masks}
  };
  std::tie(${grad_inputs}) = ${derivative};
}
""")

DERIVATIVE_TENSORLIST = CodeTemplate("""\
if (should_compute_any_outputs()) {
  grad_inputs = ${derivative};
}
""")

METHOD_DEFINITION_DERIVATIVE = CodeTemplate("""\
profiler::RecordFunction profiler("${name}");
${unpack_args}
${buffers}
${check_inplace}
${check_no_requires_grad}
std::shared_ptr<${op}> grad_fn;
auto flags = compute_flags({ ${args_with_derivatives} });
if (flags.requires_grad) {
  grad_fn = std::make_shared<${op}>(${op_ctor});
  grad_fn->is_executable = true;
  grad_fn->next_functions = compute_next_functions({ ${args_with_derivatives} });
  ${save_inputs}
}
${base_impl_call}
${no_zero_dim}
${version_counter}
${set_flags}
${save_outputs}
${record_trace}
return ${return_value};
""")

SET_FLAGS = CodeTemplate("""\
set_flags(${result}, flags, grad_fn);
""")

SET_FLAGS_INPLACE = CodeTemplate("""\
set_flags(${result}, flags, grad_fn, ${modifies_data});
""")

RECORD_TRACE = CodeTemplate("""\
if (jit::tracer::isTracing({ ${tensor_args} })) {
  jit::Node *n = jit::tracer::recordTrace( "${trace_name}", ${trace_inputs}, ${trace_outputs} );
  ${record_attributes}
}
""")

RECORD_ATTRIBUTE = CodeTemplate("""\
setattr(n, jit::stringToSymbol("${name}"), ${name});""")

CONDITIONAL = CodeTemplate("""\
if (${cond}) {
  ${statements}
}
""")

FUNCTION_PROTOTYPE = CodeTemplate("""\
${name}(${typed_args})""")

BUFFER_DECLARATION = CodeTemplate("""\
auto ${name} = tensor();
auto& ${name}_ = static_cast<VariableImpl*>(${name}.get())->data;""")

GENERATED_COMMENT = CodeTemplate("""\
generated from tools/autograd/templates/${filename}""")

template_path = os.path.join(os.path.dirname(__file__), 'templates')

VARIABLE_TYPE_H = CodeTemplate.from_file(template_path + '/VariableType.h')
VARIABLE_TYPE_CPP = CodeTemplate.from_file(template_path + '/VariableType.cpp')
FUNCTIONS_H = CodeTemplate.from_file(template_path + '/Functions.h')
FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/Functions.cpp')
PY_VARIABLE_METHODS_CPP = CodeTemplate.from_file(template_path + '/python_variable_methods.cpp')
PY_VARIABLE_DISPATCH_H = CodeTemplate.from_file(template_path + '/python_variable_methods_dispatch.h')
PY_NN_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_nn_functions.cpp')
PY_NN_FUNCTIONS_H = CodeTemplate.from_file(template_path + '/python_nn_functions.h')
PY_NN_DISPATCH_H = CodeTemplate.from_file(template_path + '/python_nn_functions_dispatch.h')
PY_FUNCTIONS_H = CodeTemplate.from_file(template_path + '/python_functions.h')
PY_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_functions.cpp')

derivatives_path = os.path.join(os.path.dirname(__file__), 'derivatives.yaml')
deprecated_path = os.path.join(os.path.dirname(__file__), 'deprecated.yaml')

# Functions with these return types delegate completely to the underlying
# base at::Type
FALLTHROUGH_RETURN_TYPES = {'int64_t', 'void*', 'bool', 'IntList'}
FALLTHROUGH_FUNCTIONS = {
    'arange', 'eye', 'linspace', 'logspace', 'tensor', 'ones', 'ones_like',
    'rand', 'randn', 'randperm', 'range', 'tensor', 'uniform', 'zeros',
    'zeros_like', 'set_',
    # these are only implemented on integral types
    '__and__', '__iand__', '__ilshift__', '__ior__', '__irshift__', '__ixor__',
    '__lshift__', '__or__', '__rshift__', '__xor__',
}
VIEW_FUNCTIONS = {
    'as_strided', 'expand', 'narrow', 'permute', 'select', 'squeeze', 't',
    'transpose', 'unfold', 'unsqueeze', 'view',
}
MANUAL_IMPLEMENTATIONS = {
    'contiguous', 'resize_', 'resize_as_'
}

# Matches "foo" in "foo, bar" but not "foobar". Used to search for the
# occurence of a parameter in the derivative formula
IDENT_REGEX = r'(^|\W){}($|\W)'


def format_return_type(returns):
    if len(returns) == 0:
        return 'void'
    elif len(returns) == 1:
        return returns[0]['type']
    else:
        return_types = [r['type'] for r in returns]
        return 'std::tuple<{}>'.format(','.join(return_types))


def write(dirname, name, template, env):
    env['generated_comment'] = GENERATED_COMMENT.substitute(filename=name)
    path = os.path.join(dirname, name)
    with open(path, 'w') as f:
        f.write(template.substitute(env))


def saved_variables(formula, args):
    # find which arguments need to be saved
    saved = []

    REPLACEMENTS = [
        # replace self.sizes() with self_sizes
        (r'{}.sizes\(\)', {
            'suffix': '_sizes',
            'type': 'IntList',
        }),
        # replace zeros_like(self) with self_info
        (r'zeros_like\({}\)', {
            'suffix': '_info',
            'type': 'TypeAndSize',
            'expr': lambda name: name,  # at save-time
            'res': lambda name: name + '_info.zeros()',  # at eval-time
        }),
        # replace self.size(2) with self_size_2
        (r'{}.size\((\w+)\)', {
            'suffix': lambda m: '_argsize_{}'.format(*m.groups()),
            'type': 'int64_t',
        }),
        # replace to_arg_sizes(self, 2) with self_argsizes_2
        (r'to_arg_sizes\({}, (\w+)\)', {
            'suffix': lambda m: '_sizes_{}'.format(*m.groups()),
            'type': 'IntList',
        }),
        # replace TensorGeometry(self) with self_geometry
        (r'TensorGeometry\({}\)', {
            'suffix': '_geometry',
            'type': 'TensorGeometry',
        }),
    ]

    for arg in args:
        if 'name' not in arg:
            # some returned arguments do not have names
            continue

        name = arg['name']

        # First search the formula for expressions which can be evaluated
        # when the autograd Function is created to avoid saving variables
        for regex, info in REPLACEMENTS:
            def repl(m):
                suffix = info['suffix']
                suffix = suffix(m) if callable(suffix) else suffix
                expr = info['expr'](name) if 'expr' in info else m.group(0)
                saved.append({
                    'name': name + suffix,
                    'type': info['type'],
                    'expr': expr,
                })
                if 'res' in info:
                    return info['res'](name)
                return name + suffix

            formula = re.sub(regex.format(name), repl, formula)

        # Find any variables which remain in the formula and save them
        if re.search(IDENT_REGEX.format(name), formula):
            arg = copy.deepcopy(arg)
            arg['type'] = arg['type'].replace('const ', '').replace(' &', '')
            saved.append(arg)

    return formula, saved


def create_derivative(declaration, formula, output_indices, var_names):
    returns = [r for r in declaration['returns'] if r.get('name') != 'self']
    arguments = declaration['arguments']
    if any(arg['name'] == 'inplace' for arg in arguments):
        for arg in arguments:
            if arg['name'] == 'input':
                returns += [arg]
        arguments = [arg for arg in arguments if arg['name'] != 'input']
    formula, saved_inputs = saved_variables(formula, arguments)
    formula, saved_outputs = saved_variables(formula, returns)

    return {
        'formula': formula,
        'output_indices': output_indices,
        'saved_inputs': saved_inputs,
        'saved_outputs': saved_outputs,
        'var_names': var_names,
    }


def create_autograd_function(name, derivatives, num_inputs, buffers=None):
    return {
        'name': name,
        'op': to_camel_case(name) + 'Backward',
        'num_inputs': num_inputs,
        'derivatives': derivatives,
        'buffers': [] if buffers is None else buffers,
        'saved_inputs': all_saved_variables(derivatives, 'saved_inputs'),
        'saved_outputs': all_saved_variables(derivatives, 'saved_outputs'),
    }


def all_saved_variables(derivatives, key):
    seen = set()
    saved = []
    for d in derivatives:
        for saved_arg in d[key]:
            if saved_arg['name'] in seen:
                continue
            seen.add(saved_arg['name'])
            saved.append(saved_arg)
    return saved


def to_camel_case(name):
    return ''.join([p.title() for p in name.split('_')])


def split_name_params(prototype):
    name, params = re.match('(\w+)\((.*)\)', prototype).groups()
    return name, params.split(', ')


def load_derivatives(path, declarations_by_signature):
    with open(path, 'r') as f:
        definitions = yaml.load(f, Loader=Loader)

    def canonical_declaration(declarations, name):
        for declaration in declarations:
            if declaration['name'] == name:
                return declaration
        # some functions only have in-place variants
        assert name + '_' == declarations[0]['name']
        return declarations[0]

    def split_names(raw_names):
        """Given "foo, bar", return ["foo", "bar"]."""
        return [x.strip() for x in raw_names.split(',')]

    def lookup_pred(pred, xs):
        """Return the index of the first element of xs matching pred."""
        return next((i, x) for i, x in enumerate(xs) if pred(x))

    # Parse each entry from derivatives.yaml
    autograd_functions = []
    for defn in definitions:
        if '(' not in defn['name']:
            continue

        # NB: Removes 'name' from defn dictionary
        defn_name, params = split_name_params(defn.pop('name'))
        param_types = [p.split(' ')[0] for p in params if p != '*']
        signature = '{}({})'.format(defn_name, ', '.join(param_types))

        declarations = declarations_by_signature[signature]
        if len(declarations) == 0:
            raise RuntimeError('no ATen declaration found for: {}'.format(signature))
        canonical = canonical_declaration(declarations, defn_name)

        # First, let us determine the set of inputs for which gradients
        # were specified in declarations.  We'll use this in layout
        # computation.
        args_with_gradients = set()
        for raw_names in defn:
            args_with_gradients |= set(split_names(raw_names))

        # Next, let us compute the layout of the grad_inputs we will
        # return.  In general this is not in one-to-one correspondence
        # with the inputs, because some will not have gradients, and we
        # will not bother allocating an undefined tensor for them.
        num_inputs = 0  # number of grad_inputs to return
        arg_name_to_output_index = {}
        for arg in canonical['arguments']:
            if arg['name'] not in args_with_gradients:
                continue
            if arg['type'] == 'TensorList':
                num_inputs = ''
                output_index = '*'  # variable length thing
            else:
                output_index = num_inputs  # the current index
                num_inputs += 1
            arg_name_to_output_index[arg['name']] = output_index

        # Finally, let us set up the derivative information
        derivatives = []
        for raw_names, formula in defn.items():
            names = split_names(raw_names)
            output_indices = []
            args = []
            for name in names:
                output_indices.append(arg_name_to_output_index[name])
                args.append(name)
            derivatives.append(create_derivative(canonical, formula, output_indices, args))

        func = create_autograd_function(defn_name, derivatives, num_inputs)
        autograd_functions.append(func)
        for declaration in declarations:
            declaration['derivative'] = func

    return autograd_functions


def ensure_unique_names(autograd_functions):
    # de-duplicate operation names
    functions_by_name = defaultdict(list)
    for func in autograd_functions:
        functions_by_name[func['op']].append(func)
    for op in functions_by_name.keys():
        overloads = functions_by_name[op]
        if len(overloads) > 1:
            for i, func in enumerate(overloads):
                func['op'] += str(i)


def preprocess_nn_functions(declarations):
    declarations_by_name = defaultdict(list)
    for d in declarations:
        declarations_by_name[d['name']].append(d)

    autograd_functions = []
    for declaration in declarations:
        name = declaration['name']
        if name == 'batch_norm' or 'conv' in name:
            continue

        fwd_name = name + '_forward'
        if fwd_name not in declarations_by_name:
            continue
        declaration['base_name'] = fwd_name

        fwd = declarations_by_name[fwd_name][0]

        input_num = 0
        bwd_name = name + '_backward'
        assert len(declarations_by_name[bwd_name]) == 1
        bwd = declarations_by_name[bwd_name][0]

        def actual(arg):
            name = arg['name']
            return name if name != 'inplace' else 'false'

        actuals = [actual(arg) for arg in bwd['arguments']]
        formula = '{}({})'.format(bwd_name, ', '.join(actuals))
        formula = formula.replace('grad_output', 'grad')
        if not re.search(IDENT_REGEX.format('grad'), formula):
            formula = '({}).mul_(grad)'.format(formula)

        # we are computing the derivatives w.r.t these variables
        var_names = []
        for ret in bwd['returns']:
            assert ret['name'].startswith('grad_')
            var_names.append(ret['name'][5:])  # remove grad_ prefix
        output_indices = list(range(len(var_names)))
        derivatives = [create_derivative(fwd, formula, output_indices, var_names)]
        input_num += len(output_indices)

        # find arguments to foo_forward() call which don't exist in foo()
        # these are buffers which have to be saved for the backwards call
        args_by_name = {arg['name']: arg for arg in declaration['arguments']}
        buffers = [arg['name'] for arg in fwd['arguments']
                   if arg['name'] not in args_by_name]

        func = create_autograd_function(name, derivatives, input_num, buffers)
        declaration['derivative'] = func
        autograd_functions.append(func)
    return autograd_functions


def create_autograd_functions(top_env, autogen_functions):
    """Functions.h and Functions.cpp body

    These contain the auto-generated subclasses of torch::autograd::Function
    for each every differentiable torch function.
    """
    function_definitions = top_env['autograd_function_definitions']
    function_declarations = top_env['autograd_function_declarations']
    py_function_initializers = top_env['py_function_initializers']

    def process_function(func):
        env = {}
        saved_variables = []
        release_variables = []
        unpack = []

        def save_arg(arg, is_output):
            name = arg['name']
            if arg['type'] == 'Tensor' or (arg['type'] == 'Scalar' and is_output):
                saved_variables.append('SavedVariable {}_;'.format(name))
                release_variables.append('{}_.data.reset();'.format(name))
                ptr = 'shared_from_this()' if is_output else ''
                unpack.append('auto {} = {}_.unpack({});'.format(name, name, ptr))
            elif arg['type'] == 'IntList':
                saved_variables.append('std::vector<int64_t> {};'.format(name))
            else:
                saved_variables.append('{} {};'.format(arg['type'], name))

        for arg in func['saved_inputs']:
            save_arg(arg, is_output=False)
        for arg in func['saved_outputs']:
            save_arg(arg, is_output=True)
        env['saved_variables'] = saved_variables
        env['release_variables'] = release_variables

        def uses_grad(func):
            for derivative in func['derivatives']:
                formula = derivative['formula']
                if re.search(IDENT_REGEX.format('grad'), formula):
                    return True
            return False

        body = []

        if uses_grad(func):
            body.append('auto& grad = grads[0];')

        def emit_derivative(derivative):
            formula = derivative['formula']
            idxs = derivative['output_indices']
            if idxs == ['*']:
                return DERIVATIVE_TENSORLIST.substitute(derivative=formula)
            elif len(idxs) == 1:
                return DERIVATIVE_TENSOR.substitute(idx=idxs[0], derivative=formula)
            else:
                grad_inputs = ', '.join(['grad_inputs[{}]'.format(i) for i in idxs])
                masks = ['should_compute_output({}),'.format(i) for i in idxs]
                return DERIVATIVE_MULTI.substitute(
                    idxs=idxs, derivative=formula, grad_inputs=grad_inputs,
                    masks=masks, n=len(idxs))

        body.extend(unpack)
        for derivative in func['derivatives']:
            body.append(emit_derivative(derivative))

        env['body'] = body
        env = nested_dict(env, func)
        function_declarations.append(FUNCTION_DECLARATION.substitute(env))
        function_definitions.append(FUNCTION_DEFINITION.substitute(env))
        py_function_initializers.append(PY_FUNCTION_DEFINITION.substitute(env))

    for func in autogen_functions:
        process_function(func)


def is_implemented(option):
    return (option['return_type'] in FALLTHROUGH_RETURN_TYPES or
            option['name'] in FALLTHROUGH_FUNCTIONS or
            option['name'].endswith('_backward') or
            option.get('derivative') is not None)


def create_variable_type(top_env, aten_declarations):
    """VariableType.h and VariableType.cpp body

    This is the at::Type subclass for differentiable tensors. The
    implementation of each function dispatches to the base tensor type to
    compute the output. The grad_fn is attached to differentiable functions.
    """

    type_declarations = top_env['type_derived_method_declarations']
    type_definitions = top_env['type_derived_method_definitions']

    def skip_function(declaration):
        name = declaration['name']
        return name.endswith('_out') or name.endswith('_forward')

    def find_args_with_derivatives(func, tensor_arg_names):
        """Find arguments that have derivative definitions"""
        names = set(name for d in func['derivatives'] for name in d['var_names'])
        differentiable = [arg for arg in tensor_arg_names if arg in names]
        if len(differentiable) != len(names):
            missing = names - set(differentiable)
            raise RuntimeError('Missing arguments for derivatives: {}'.format(missing))
        return differentiable

    def save_variables(option, saved_variables, is_output):
        # assign the saved variables to the generated grad_fn
        stmts = []
        for arg in saved_variables:
            name = arg['name']
            expr = arg.get('expr', arg['name'])
            if is_output and not option['inplace']:
                if len(option['returns']) > 1:
                    # unpack multiple outputs
                    return_names = [r['name'] for r in option['returns']]
                    idx = return_names.index(name)
                    stmts.append('auto& {} = std::get<{}>(ret);'.format(name, idx))
                elif name != 'input':
                    stmts.append('auto& {} = ret;'.format(name))
            if arg['type'] == 'Tensor' or (is_output and arg['type'] == 'Scalar'):
                name += '_'
                var = arg['name']
                if var == 'self' and option['inplace']:
                    var = 'self.clone()'
                    assert not is_output
                if option['inplace'] and is_output:
                    var = 'self'
                ptr = 'grad_fn.get()' if is_output else 'nullptr'
                expr = 'SavedVariable({}, {})'.format(var, ptr)
            stmts.append('grad_fn->{} = {};'.format(name, expr))
        return stmts

    def requires_unpack(arg):
        return 'Tensor' in arg['dynamic_type']

    def get_suffix(dynamic_type, is_nullable):
        if is_nullable:
            assert dynamic_type == 'Tensor'
            return '_opt'
        elif dynamic_type == 'IndexTensor':
            return '_long'
        elif dynamic_type == 'BoolTensor':
            return '_byte'
        else:
            return ''

    def unpack_args(env, option):
        body = []
        unpacked_args = []
        for i, arg in enumerate(option['arguments']):
            if not requires_unpack(arg):
                unpacked_args.append(arg['name'])
                continue

            dynamic_type = arg['dynamic_type']
            is_nullable = arg.get('is_nullable', False)
            ref = (not is_nullable) and dynamic_type != 'TensorList'
            suffix = get_suffix(dynamic_type, is_nullable)

            body.append(UNPACK_TENSOR.substitute(
                arg_name=arg['name'],
                arg_pos=i,
                suffix=suffix,
                ref='&' if ref else '',
            ))
            unpacked_args.append(arg['name'] + '_')

        if option.get('derivative') is not None:
            for arg in option['derivative'].get('buffers', []):
                unpacked_args.append(arg + '_')
        env['unpacked_args'] = unpacked_args
        return body

    def emit_buffers(buffers):
        res = []
        for name in buffers:
            res.append(BUFFER_DECLARATION.substitute(name=name))
        return res

    def emit_record_trace(env, declaration):

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

        arguments = declaration['arguments']
        tensor_args = [arg for arg in arguments if arg['simple_type'] in {'Tensor', 'TensorList'}]
        if len(tensor_args) == 1 and tensor_args[0]['simple_type'] == 'TensorList':
            # Special case for TensorList.  This only works when there
            # is a single argument
            local['trace_inputs'] = "cast_tensor_list({})".format(declaration['arguments'][0]['name'])
        else:
            local['trace_inputs'] = CodeTemplate("{ ${tensor_args} }").substitute(env)

        local['record_attributes'] = []
        for arg in declaration['arguments']:
            if arg['simple_type'] in {'Tensor', 'TensorList'}:
                continue
            local['record_attributes'].append(RECORD_ATTRIBUTE.substitute(name=arg['name']))
        if not local['record_attributes']:
            local['record_attributes'].append('(void)n;')

        local['trace_name'] = declaration['api_name']
        if local['trace_name'].endswith('_'):
            local['trace_name'] = local['trace_name'][:-1]

        combined = nested_dict(local, nested_dict(env, declaration))
        return RECORD_TRACE.substitute(combined)

    def emit_check_no_requires_grad(tensor_args, args_with_derivatives):
        """Checks that arguments without derivatives don't require grad"""
        body = []
        for arg in tensor_args:
            name = arg['name']
            if name in args_with_derivatives:
                continue
            if name == 'output':
                # Double-backwards definitions sometimes take in 'input' and
                # 'output', but only define the derivative for input.
                continue
            if arg['dynamic_type'] in {'IndexTensor', 'BoolTensor'}:
                continue
            body.append('check_no_requires_grad({}, "{}");'.format(name, name))
        return body

    def emit_body(declaration):
        if not is_implemented(declaration):
            return METHOD_DEFINITION_NYI.substitute(declaration)

        env = {}
        body = []
        env['unpack_args'] = unpack_args(env, declaration)

        combined = nested_dict(env, declaration)
        if declaration['return_type'] in FALLTHROUGH_RETURN_TYPES:
            body.extend(METHOD_DEFINITION_FALLTHROUGH.substitute(combined).split('\n'))
            return body
        elif declaration['name'] in FALLTHROUGH_FUNCTIONS:
            tmpl = (METHOD_DEFINITION_FALLTHROUGH_INPLACE if declaration['inplace']
                    else METHOD_DEFINITION_FALLTHROUGH_VARIABLE)
            body.extend(tmpl.substitute(combined).split('\n'))
            return body

        arguments = declaration['arguments']
        tensor_args = [arg for arg in arguments if arg['simple_type'] in {'Tensor', 'TensorList'}]
        env['tensor_args'] = [arg['name'] for arg in tensor_args]

        name = declaration['name']
        base_name = name[:-1] if declaration['inplace'] else name
        is_view = base_name in VIEW_FUNCTIONS

        if declaration['inplace']:
            env['return_value'] = 'self'
            env['trace_outputs'] = '{ self }'
            env['result'] = 'static_cast<Variable&>(self)'
        elif declaration['return_type'] == 'std::vector<Tensor>':
            env['return_value'] = 'as_tensor_list(ret)'
            env['result'] = 'ret'
            env['trace_outputs'] = 'ret'
        else:
            env['return_value'] = '{}(std::move(ret))'.format(declaration['return_type'])
            env['result'] = 'std::get<0>(ret)' if len(declaration['returns']) > 1 else 'ret'
            if len(declaration['returns']) > 1:
                # NB: This won't work if we get heterogenous outputs
                outs = ['std::get<{}>(ret)'.format(i)
                        for i, v in enumerate(declaration['returns']) if v['type'] == 'Tensor']
            else:
                outs = ['ret']
            env['trace_outputs'] = CodeTemplate("{ ${outs} }").substitute(outs=outs)

        if any(arg['simple_type'] in {'Generator', 'Storage'} for arg in arguments):
            env['record_trace'] = []
        else:
            env['record_trace'] = emit_record_trace(env, declaration)

        func = declaration.get('derivative')
        if func is not None:
            env['op'] = func['op']
            env['op_ctor'] = ''
            env['buffers'] = emit_buffers(func.get('buffers', []))
            env['save_inputs'] = save_variables(declaration, func['saved_inputs'], False)
            env['save_outputs'] = save_variables(declaration, func['saved_outputs'], True)
            env['args_with_derivatives'] = find_args_with_derivatives(func, env['tensor_args'])
        else:
            env['op'] = 'Error'
            env['op_ctor'] = '"the derivative for {} is not implemented"'.format(declaration['api_name'])
            env['buffers'] = []
            env['save_inputs'] = []
            env['save_outputs'] = []
            env['args_with_derivatives'] = env['tensor_args']

        env['check_no_requires_grad'] = emit_check_no_requires_grad(
            tensor_args, env['args_with_derivatives'])
        if len(env['save_outputs']) > 0:
            env['save_outputs'] = CONDITIONAL.substitute(
                cond='grad_fn', statements=env['save_outputs'])

        env['check_inplace'] = ''
        env['version_counter'] = ''
        env['no_zero_dim'] = ''
        maybe_inplace = any(arg['name'] == 'inplace' for arg in arguments)
        base_call = BASE_CALL.substitute(combined)
        if declaration['inplace']:
            env['check_inplace'] = 'check_inplace(self);'
            env['version_counter'] = 'increment_version(self);'
            modifies_data = 'false' if is_view else 'true'
            env['set_flags'] = SET_FLAGS_INPLACE.substitute(combined, modifies_data=modifies_data)
            if is_view:
                env['no_zero_dim'] = 'ensure_no_aten_scalars(self);'
        elif maybe_inplace:
            assert not is_view, declaration['name']
            env['check_inplace'] = 'if (inplace) check_inplace(input);'
            env['version_counter'] = 'if (inplace) increment_version(input);'
            env['set_flags'] = SET_FLAGS_INPLACE.substitute(combined, modifies_data='inplace')
            base_call = 'auto ret = maybe_wrap({}, input, inplace)'.format(base_call)
        else:
            env['set_flags'] = SET_FLAGS.substitute(combined)
            if is_view:
                base_call = 'auto ret = as_view(static_cast<const Variable&>(self), {})'.format(base_call)
            else:
                base_call = 'auto ret = as_variable({})'.format(base_call)

        env['base_impl_call'] = base_call + ';'

        body.extend(METHOD_DEFINITION_DERIVATIVE.substitute(combined).split('\n'))
        return body

    def process_function(declaration):
        if skip_function(declaration):
            return

        if declaration.get('derivative') is None and declaration['mode'] == 'native':
            # native functions without a derivative don't need Type implementations
            return

        env = {}
        env['type_definition_body'] = emit_body(declaration)

        combined = nested_dict(env, declaration)
        if 'Type' in combined['method_of']:
            type_declarations.append(METHOD_DECLARATION.substitute(combined))
            if declaration['name'] not in MANUAL_IMPLEMENTATIONS:
                type_definitions.append(METHOD_DEFINITION.substitute(combined))

    for declaration in aten_declarations:
        process_function(declaration)


def load_aten_declarations(path):
    with open(path, 'r') as f:
        declarations = yaml.load(f, Loader=Loader)

    # enrich declarations with additional information
    for declaration in declarations:
        for arg in declaration['arguments']:
            simple_type = arg['type']
            simple_type = simple_type.replace(' &', '').replace('const ', '')
            simple_type = simple_type.replace('Generator *', 'Generator')
            arg['simple_type'] = simple_type
        declaration['formals'] = [arg['type'] + ' ' + arg['name']
                                  for arg in declaration['arguments']]
        declaration['args'] = [arg['name'] for arg in declaration['arguments']]
        declaration['api_name'] = declaration['name']
        declaration['return_type'] = format_return_type(declaration['returns'])

        declaration['base_name'] = declaration['name']

        # if the return value is missing a name, call it 'output'
        for ret in declaration['returns']:
            if 'name' not in ret:
                assert len(declaration['returns']) == 1
                ret['name'] = 'result'

        # Compute the Python function prototype for argument parsing
        typed_args = []
        positional = True
        for arg in declaration['arguments']:
            if arg.get('kwarg_only', False) and positional:
                typed_args.append('*')
                positional = False
            typename = arg['simple_type']
            if arg.get('size') is not None:
                typename = '{}[{}]'.format(typename, arg['size'])
            param = typename + ' ' + arg['name']
            if arg.get('default') is not None:
                default = arg['default']
                if default == 'nullptr' or default == '{}':
                    default = 'None'
                param += '=' + str(default)
            typed_args.append(param)

        # Python function prototype
        declaration['typed_args'] = typed_args
        declaration['prototype'] = FUNCTION_PROTOTYPE.substitute(declaration)

    return declarations


def load_deprecated_signatures(declarations_by_signature):
    with open(deprecated_path, 'r') as f:
        deprecated_defs = yaml.load(f, Loader=Loader)
    declarations = []

    def get_signature(name, params, call_args):
        # create a mapping of parameter name to parameter type
        types = dict([param.split(' ')[::-1] for param in params])
        # if the name in the call is not in the parameter list, assume it's
        # a literal Scalar
        rearranged_types = [types.get(arg, 'Scalar') for arg in call_args]
        return '{}({})'.format(name, ', '.join(rearranged_types))

    for deprecated in deprecated_defs:
        prototype = deprecated['name']
        call_args = split_name_params(deprecated['aten'])[1]
        name, params = split_name_params(prototype)
        signature = get_signature(name, params, call_args)

        for declaration in declarations_by_signature[signature]:
            declaration = copy.deepcopy(declaration)
            declaration['deprecated'] = True
            declaration['call_args'] = call_args
            if declaration['inplace']:
                declaration['prototype'] = prototype.replace(name, name + '_')
            else:
                declaration['prototype'] = prototype

            args_by_name = {arg['name']: arg for arg in declaration['arguments']}
            declaration['arguments'] = []
            for arg in params:
                _, arg_name = arg.split(' ')
                declaration['arguments'].append(args_by_name[arg_name])
            declarations.append(declaration)
    return declarations


def gen_variable_type(declarations, out):
    aten_decls = load_aten_declarations(declarations)

    def group_declarations_by_signature():
        d = defaultdict(list)
        for declaration in aten_decls:
            name = declaration['name']
            base_name = name[:-1] if declaration['inplace'] else name
            simple_types = [arg['simple_type'] for arg in declaration['arguments']]
            signature = '{}({})'.format(base_name, ', '.join(simple_types))
            d[signature].append(declaration)
        return d

    declarations_by_signature = group_declarations_by_signature()

    th_autograd_funcs = load_derivatives(derivatives_path, declarations_by_signature)
    nn_autograd_funcs = preprocess_nn_functions(aten_decls)
    all_autograd_functions = th_autograd_funcs + nn_autograd_funcs
    ensure_unique_names(all_autograd_functions)

    def should_generate_python_binding(declaration):
        name = declaration['name']
        # don't bind (non-native) unimplemented functions to prevent errors in test_autograd.
        # Native functions, even if they don't have derivatives specified, should be bound
        # so they can be called from python (their derivatives are defined based on the functions
        # they call).
        if not is_implemented(declaration) and declaration['mode'] != 'native':
            return False

        # don't bind size or stride since the python signatures are different
        if name in ['size', 'stride']:
            return False

        if name.endswith('_backward'):
            return False

        # we don't currently support functions which are only defined on Type
        # such as zeros(), randn(), etc.
        method_of = declaration['method_of']
        if 'Tensor' not in method_of and 'namespace' not in method_of:
            return False

        return True

    py_variable_methods = defaultdict(list)
    py_nn_functions = defaultdict(list)
    for declaration in aten_decls:
        name = declaration['name']
        if not should_generate_python_binding(declaration):
            continue
        if declaration['mode'] == 'NN':
            py_nn_functions[name].append(declaration)
        else:
            py_variable_methods[name].append(declaration)

    for declaration in load_deprecated_signatures(declarations_by_signature):
        py_variable_methods[declaration['name']].append(declaration)

    env = {
        'autograd_function_declarations': [],
        'autograd_function_definitions': [],
        'type_derived_method_declarations': [],
        'type_derived_method_definitions': [],
        'py_methods': [],
        'py_method_defs': [],
        'py_method_dispatch': [],
        'py_function_initializers': [],
        'py_nn_functions': [],
        'py_nn_function_defs': [],
        'py_nn_function_dispatch': [],
    }

    create_autograd_functions(env, all_autograd_functions)
    create_variable_type(env, aten_decls)

    from .gen_python_functions import create_python_bindings
    create_python_bindings(
        py_variable_methods,
        env['py_methods'],
        env['py_method_defs'],
        env['py_method_dispatch'],
        is_class=True)

    create_python_bindings(
        py_nn_functions,
        env['py_nn_functions'],
        env['py_nn_function_defs'],
        env['py_nn_function_dispatch'],
        is_class=False)

    write(out, 'VariableType.h', VARIABLE_TYPE_H, env)
    write(out, 'VariableType.cpp', VARIABLE_TYPE_CPP, env)
    write(out, 'Functions.h', FUNCTIONS_H, env)
    write(out, 'Functions.cpp', FUNCTIONS_CPP, env)
    write(out, 'python_variable_methods.cpp', PY_VARIABLE_METHODS_CPP, env)
    write(out, 'python_variable_methods_dispatch.h', PY_VARIABLE_DISPATCH_H, env)
    write(out, 'python_nn_functions.cpp', PY_NN_FUNCTIONS_CPP, env)
    write(out, 'python_nn_functions.h', PY_NN_FUNCTIONS_H, env)
    write(out, 'python_nn_functions_dispatch.h', PY_NN_DISPATCH_H, env)
    write(out, 'python_functions.h', PY_FUNCTIONS_H, env)
    write(out, 'python_functions.cpp', PY_FUNCTIONS_CPP, env)


def main():
    parser = argparse.ArgumentParser(
        description='Generate autograd C++ files script')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    args = parser.parse_args()
    gen_variable_type(args.declarations, args.out)


if __name__ == '__main__':
    main()
