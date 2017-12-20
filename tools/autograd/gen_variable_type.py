# gen_variable_type.py's primary purpose is to generate
# VariableType.cpp, which provides the binding code necessary to provide
# a differentiable version of ATen operators.  There are a number of
# different things we could mean:
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
#     These are currently called "fallthrough" functions, although
#     they are not entirely fallthrough; for example, if the function
#     in question returns a tensor, we have to wrap it in a
#     (does not require grad) variable to abide by the API contract
#     of Variable.
#
#   - Some function don't need a backwards implementation because they
#     are implement as a composition of other (differentiable) ATen
#     functions.  These are dispatched directly to the Type superclass,
#     which will in turn dispatch back to VariableType for its
#     differentiable subcomponents.

import argparse
import copy
import os
import re
import yaml
import warnings
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
virtual ${return_type} ${method_prefix_derived}${api_name}(${formals}) const override;
""")

METHOD_DEFINITION = CodeTemplate("""\
${return_type} VariableType::${method_prefix_derived}${api_name}(${formals}) const {
    ${type_definition_body}
}
""")

METHOD_DEFINITION_NYI = CodeTemplate("""\
throw std::runtime_error("VariableType::${api_name} NYI");""")

DERIVED_CALL = CodeTemplate("""\
baseType->${method_prefix_derived}${base_name}(${unpacked_args})""")

UNPACK_TENSOR = CodeTemplate("""\
auto${ref} ${arg_name}_ = unpack${suffix}(${arg_name}, "${arg_name}", ${arg_pos});""")

FUNCTION_DECLARATION = CodeTemplate("""\
struct ${op} : public ${superclass} {
  using ${superclass}::${superclass};
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
  auto grad_input_mask = std::array<bool, ${n}>{
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

# NB: both fallthrough and derivative dispatch via derived (aka baseType).
# That is why all of these paths need unpack_args.

METHOD_DEFINITION_BODY_FALLTHROUGH = CodeTemplate("""\
${unpack_args}
return baseType->${method_prefix_derived}${api_name}(${unpacked_args});""")

METHOD_DEFINITION_BODY_FALLTHROUGH_VARIABLE = CodeTemplate("""\
${unpack_args}
return as_variable(baseType->${method_prefix_derived}${api_name}(${unpacked_args}));""")

METHOD_DEFINITION_BODY_FALLTHROUGH_INPLACE = CodeTemplate("""\
${unpack_args}
baseType->${method_prefix_derived}${api_name}(${unpacked_args});
increment_version(self);
return self;
""")

# XXX: the order of definitions here is actually very important, even when it's not
# visible at first glance. The non-obvious invariants are:
# - set_flags has to appear after version_counter, because rebase_history requires
#   that the counter is incremented before it is called
# - record_trace has to appear before save_outputs, because the saved variables
#   should have their trace set correctly
METHOD_DEFINITION_BODY_DERIVATIVE = CodeTemplate("""\
profiler::RecordFunction profiler("${name}");
${unpack_args}
${buffers}
${check_inplace}
${check_no_requires_grad}
std::shared_ptr<${op}> grad_fn;
auto requires_grad = compute_requires_grad({ ${args_with_derivatives} });
if (requires_grad) {
  grad_fn = std::make_shared<${op}>(${op_ctor});
  grad_fn->next_functions = compute_next_functions({ ${args_with_derivatives} });
  ${save_inputs}
}
${base_impl_call}
${no_zero_dim}
${version_counter}
${set_flags}
${record_trace}
${save_outputs}
return ${return_value};
""")

METHOD_DEFINITION_BODY_VIA_TYPE = CodeTemplate("""\
profiler::RecordFunction profiler("${name}");
auto ret = Type::${method_prefix_derived}${api_name}(${args});
${record_trace}
return ${return_value};
""")

SET_HISTORY = CodeTemplate("""\
set_history(${result}, grad_fn);
""")

REBASE_HISTORY = CodeTemplate("""\
rebase_history(${result}, grad_fn);
""")

RECORD_TRACE = CodeTemplate("""\
if (jit::tracer::isTracing( ${tensor_args} )) {
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
    'rand', 'randn', 'randperm', 'range', 'tensor', 'zeros',
    'zeros_like', 'set_', '_indices', '_values',
    # these are only implemented on integral types
    '__and__', '__iand__', '__ilshift__', '__ior__', '__irshift__', '__ixor__',
    '__lshift__', '__or__', '__rshift__', '__xor__',
}
VIEW_FUNCTIONS = {
    'alias', 'as_strided', 'expand', 'narrow', 'permute', 'select', 'slice',
    'squeeze', 't', 'transpose', 'unfold', 'unsqueeze', 'view',
}
MANUAL_IMPLEMENTATIONS = {
    'contiguous', 'resize_', 'resize_as_'
}
# These functions require manual Python bindings or are not exposed to Python
SKIP_PYTHON_BINDINGS = [
    'alias', 'contiguous', 'clamp.*', 'is_cuda', 'size', 'stride',
    '.*_backward'
]
# These functions have backwards which cannot be traced, and so must have
# their backward functions traced opaquely.
# VIEW_FUNCTIONS are not traceable because they use as_strided, which
# has an untraceable backwards, see
# https://github.com/pytorch/pytorch/issues/4250
# TODO: This is probably not exhaustive, but it's a start
UNTRACEABLE_FUNCTIONS = VIEW_FUNCTIONS
# These functions we don't want to record for tracing, because we always want
# to trace their constituent parts.  This is a temporary hack in lieue
# of proper scopes, where subsequent compilation passes can ask for the unfolding
# on demand.  Only concrete ATen methods can be disabled this way; it will have
# NO EFFECT otherwise.
DONT_RECORD_TRACE = {'convolution', 'conv1d', 'conv2d', 'conv3d',
                     'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d'}

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


# TODO: Use a real parser here; this will get bamboozled
# by signatures that contain things like std::array<bool, 2> (note the space)
def split_name_params(prototype):
    name, params = re.match('(\w+)\((.*)\)', prototype).groups()
    return name, params.split(', ')


def load_derivatives(path, declarations_by_signature, declarations_by_name):
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

    def set_up_derivatives(defn, declaration):

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
        for arg in declaration['arguments']:
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
        for raw_names in sorted(defn.keys()):
            formula = defn[raw_names]
            names = split_names(raw_names)
            output_indices = []
            args = []
            for name in names:
                output_indices.append(arg_name_to_output_index[name])
                args.append(name)
            derivatives.append(create_derivative(declaration, formula, output_indices, args))

        return derivatives, num_inputs

    def is_nn_fwd(defn_name, declarations_by_name):
        """Return True if the definition is of an NN, non-double
           backward function, False otherwise"""

        if len(declarations_by_name[defn_name]) == 0:
            return False
        declaration = declarations_by_name[defn_name][0]
        base_name = defn_name if not declaration['inplace'] else defn_name[:-1]
        fwd_name = base_name + '_forward'
        if declaration['mode'] != 'NN' or fwd_name not in declarations_by_name:
            return False
        return True

    def preprocess_nn_function(defn_name, declarations_by_name):
        """Set up declaration and derivative information for NN,
           non-double backward functions"""

        declaration = declarations_by_name[defn_name][0]
        base_name = defn_name if not declaration['inplace'] else defn_name[:-1]
        fwd_name = base_name + ('_forward' if not declaration['inplace'] else '_forward_')

        assert len(declarations_by_name[fwd_name]) == 1

        declaration['base_name'] = fwd_name
        fwd = declarations_by_name[fwd_name][0]

        derivatives, num_inputs = set_up_derivatives(defn, fwd)
        buffers = declaration['buffers']

        func = create_autograd_function(defn_name, derivatives, num_inputs, buffers)
        declaration['derivative'] = func

        return func

    # Parse each entry from derivatives.yaml
    autograd_functions = []
    for defn in definitions:
        if '(' not in defn['name']:
            continue

        def unzip(xs):
            return zip(*xs)

        # NB: Removes 'name' from defn dictionary
        defn_name, params = split_name_params(defn.pop('name'))
        param_types, param_names = unzip([p.split(' ') for p in params if p != '*'])
        if 'grad_input_mask' in param_names:
            raise RuntimeError("Signature for {} has an argument named grad_input_mask, "
                               "but this name would be shadowed by our codegen. "
                               "Please use a different name in Declarations.cwrap."
                               .format(defn_name))
        signature = '{}({})'.format(defn_name, ', '.join(param_types))

        if is_nn_fwd(defn_name, declarations_by_name):
            func = preprocess_nn_function(defn_name, declarations_by_name)
            if func is not None:
                autograd_functions.append(func)
            continue

        declarations = declarations_by_signature[signature]
        if len(declarations) == 0:
            avail = [k for k, v in declarations_by_signature.items()
                     if k.startswith(defn_name + '(') and len(v) > 0]
            raise RuntimeError('no ATen declaration found for: {}.  '
                               'Available signatures: {}'.format(signature, ', '.join(avail)))
        canonical = canonical_declaration(declarations, defn_name)

        # TODO: Check the types line up
        if len(param_names) != len(canonical['args']):
            raise RuntimeError('Signature for {} has {} arguments ({}), but '
                               'Declarations.yaml records {} arguments ({})'
                               .format(defn_name,
                                       len(param_names),
                                       ', '.join(param_names),
                                       len(canonical['args']),
                                       ', '.join(canonical['args'])))
        for i, (x, y) in enumerate(zip(param_names, canonical['args'])):
            if x != y:
                raise RuntimeError('Argument {} of {} has different names in '
                                   'derivatives.yaml ({}) and '
                                   'Declarations.yaml ({})'
                                   .format(i, defn_name, x, y))

        derivatives, num_inputs = set_up_derivatives(defn, canonical)
        buffers = canonical.get('buffers')

        func = create_autograd_function(defn_name, derivatives, num_inputs, buffers)
        autograd_functions.append(func)
        for declaration in declarations:
            declaration['derivative'] = func

    return autograd_functions


def ensure_unique_names(autograd_functions):
    # de-duplicate operation names
    # you end up with something like:
    #   AddBackward0
    #   AddBackward1
    # one for each overload
    functions_by_name = defaultdict(list)
    for func in autograd_functions:
        functions_by_name[func['op']].append(func)
    for op in functions_by_name.keys():
        overloads = functions_by_name[op]
        if len(overloads) > 1:
            for i, func in enumerate(overloads):
                func['op'] += str(i)


def uses_grad(func):
    if func is None:
        return False
    for derivative in func['derivatives']:
        formula = derivative['formula']
        if re.search(IDENT_REGEX.format('grad'), formula):
            return True
    return False


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
        if func['name'] in UNTRACEABLE_FUNCTIONS:
            env['superclass'] = 'Function'
        else:
            env['superclass'] = 'TraceableFunction'
        env = nested_dict(env, func)
        function_declarations.append(FUNCTION_DECLARATION.substitute(env))
        function_definitions.append(FUNCTION_DEFINITION.substitute(env))
        py_function_initializers.append(PY_FUNCTION_DEFINITION.substitute(env))

    for func in autogen_functions:
        process_function(func)


def dispatch_strategy(declaration):
    """How are we going to call the underlying implementation of a
    declaration?  There are three strategies:

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

        - unimplemented: we don't have an underlying implementation, so
          we will immediately error if they call this method on VariableType.
    """
    def use_derived(option):
        return (option['return_type'] in FALLTHROUGH_RETURN_TYPES or
                option['name'] in FALLTHROUGH_FUNCTIONS or
                option['name'] in MANUAL_IMPLEMENTATIONS or
                # TODO: Now that NN is represented explicitly in
                # derivatives.yaml, get rid of this test soon.  We can't get
                # rid of it /quite/ yet because we need to add NYI entries
                # to derivatives.yaml, but actually they'll get real entries
                # in https://github.com/pytorch/pytorch/pull/4116 so I am
                # too lazy to delete it now.
                option['name'].endswith('_backward') or
                option.get('derivative') is not None)

    if use_derived(declaration):
        return 'use_derived'
    elif not declaration['abstract']:
        # This applies a heuristic: if the function is concrete (we
        # don't have to override it) and we didn't declare it in
        # derivatives.yaml, we'll assume that it is actually implemented
        # out of differentiable functions.  (This assumption might not
        # hold, but then you'll see gradcheck fail.)
        return 'use_type'
    else:
        return 'unimplemented'


def create_variable_type(top_env, aten_declarations):
    """VariableType.h and VariableType.cpp body

    This is the at::Type subclass for differentiable tensors. The
    implementation of each function dispatches to the base tensor type to
    compute the output. The grad_fn is attached to differentiable functions.
    """

    type_declarations = top_env['type_derived_method_declarations']
    type_definitions = top_env['type_derived_method_definitions']

    declarations_by_name = defaultdict(list)
    for d in aten_declarations:
        declarations_by_name[d['name']].append(d)

    def skip_function(declaration):
        name = declaration['name']
        return name.endswith('_out') or '_forward' in name

    def find_args_with_derivatives(func, tensor_arg_names):
        """Find arguments that have derivative definitions"""
        names = set(name for d in func['derivatives'] for name in d['var_names'])
        differentiable = [arg for arg in tensor_arg_names if arg in names]
        if len(differentiable) != len(names):
            missing = names - set(differentiable)
            raise RuntimeError('Missing arguments for derivatives: {} in {}'.format(missing, func['name']))
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
                expr = 'SavedVariable({}, {})'.format(var, str(is_output).lower())
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

    def unpack_args(env, declaration):
        body = []
        unpacked_args = []
        for i, arg in enumerate(declaration['arguments']):
            if not requires_unpack(arg):
                unpacked_args.append(arg['name'])
                continue

            dynamic_type = arg['dynamic_type']
            is_nullable = arg.get('is_nullable', False)
            ref = (not is_nullable) and dynamic_type not in ['TensorList', 'SparseTensor']
            suffix = get_suffix(dynamic_type, is_nullable)
            if dynamic_type == 'TensorList' and declaration['name'] == 'index':
                # TODO: specify this in Declarations.yaml somehow
                suffix = '_idxs'

            body.append(UNPACK_TENSOR.substitute(
                arg_name=arg['name'],
                arg_pos=i,
                suffix=suffix,
                ref='&' if ref else '',
            ))
            unpacked_args.append(arg['name'] + '_')

        if declaration.get('derivative') is not None:
            for arg in declaration['derivative'].get('buffers', []):
                unpacked_args.append(arg + '_')
        env['unpacked_args'] = unpacked_args
        return body

    def emit_buffers(buffers):
        res = []
        for name in buffers:
            res.append(BUFFER_DECLARATION.substitute(name=name))
        return res

    def mk_tuple_getters(declaration, pred):
        # NB: This won't work if we get heterogenous outputs
        return ['std::get<{}>(ret)'.format(i)
                for i, v in enumerate(declaration['returns'])
                if v['type'] == 'Tensor' and pred(v)]

    def get_trace_outputs(declaration):
        if len(declaration['returns']) > 1:
            trace_outs = mk_tuple_getters(declaration, lambda v: True)
        else:
            trace_outs = ['ret']
        return CodeTemplate("{ ${outs} }").substitute(outs=trace_outs)

    def emit_record_trace(env, declaration):

        # Operations involving Generator and Storage are not traceable
        # at the moment
        if any(arg['simple_type'] in {'Generator', 'Storage'} for arg in declaration['arguments']):
            return []

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
        if any(arg['simple_type'] == 'TensorList' for arg in tensor_args):
            # Allocate a temporary vector with flatten and pass it in
            local['trace_inputs'] = CodeTemplate("flatten( $tensor_args )").substitute(env)
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

    def emit_body_via_type(declaration):
        env = {}
        body = []

        combined = nested_dict(env, declaration)
        arguments = declaration['arguments']

        tensor_args = [arg for arg in arguments if arg['simple_type'] in {'Tensor', 'TensorList'}]
        env['tensor_args'] = [arg['name'] for arg in tensor_args]

        if declaration['inplace']:
            env['return_value'] = 'self'
            env['trace_outputs'] = '{ self }'
        elif declaration['return_type'] == 'std::vector<Tensor>':
            env['return_value'] = 'ret'
            env['trace_outputs'] = 'cast_tensor_list(ret)'
        else:
            env['return_value'] = '{}(std::move(ret))'.format(declaration['return_type'])
            env['trace_outputs'] = get_trace_outputs(declaration)

        if declaration['name'] in DONT_RECORD_TRACE:
            env['record_trace'] = ''
        else:
            env['record_trace'] = emit_record_trace(env, declaration)

        body.extend(METHOD_DEFINITION_BODY_VIA_TYPE.substitute(combined).split('\n'))
        return body

    def emit_body_via_derived(declaration):
        env = {}
        body = []
        env['unpack_args'] = unpack_args(env, declaration)

        combined = nested_dict(env, declaration)
        if declaration['return_type'] in FALLTHROUGH_RETURN_TYPES:
            body.extend(METHOD_DEFINITION_BODY_FALLTHROUGH.substitute(combined).split('\n'))
            return body
        elif declaration['name'] in FALLTHROUGH_FUNCTIONS:
            tmpl = (METHOD_DEFINITION_BODY_FALLTHROUGH_INPLACE if declaration['inplace']
                    else METHOD_DEFINITION_BODY_FALLTHROUGH_VARIABLE)
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
            if len(declaration['returns']) > 1:
                diff_outs = mk_tuple_getters(declaration, lambda v: v['dynamic_type'] == 'Tensor')
            else:
                diff_outs = ['ret']
            # TODO: This is a bit dodgy, but the basic idea is, if you
            # used 'grad' in the derivative computation, you have
            # implicitly assumed that there is only one gradient being
            # passed into you, which in turn means that only the first
            # return of the corresponding forward was meant to be
            # differentiable.  If you actually wanted to differentiate
            # on the other returns, you would have used 'grads' instead.
            # This happens in practice with 'gesv'.
            #
            # I don't think this is a good way to implement this, but
            # there doesn't seem to be a good place to mark things as
            # differentiable or non-differentiable at the moment.
            if uses_grad(declaration.get('derivative')):
                env['result'] = "std::get<0>(ret)" if len(declaration['returns']) > 1 else 'ret'
            else:
                env['result'] = CodeTemplate("{ ${outs} }").substitute(outs=diff_outs)
            env['trace_outputs'] = get_trace_outputs(declaration)

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
        base_call = DERIVED_CALL.substitute(combined)
        if declaration['inplace']:
            env['check_inplace'] = 'check_inplace(self);'
            env['version_counter'] = 'increment_version(self);'
            if is_view:
                # in-place view functions like squeeze_() go through a different
                # code path because these functions only affect the tensor on
                # which they're called, not other views of the same data.
                env['set_flags'] = SET_HISTORY.substitute(combined)
            else:
                env['set_flags'] = REBASE_HISTORY.substitute(combined)
            if is_view:
                env['no_zero_dim'] = 'ensure_no_aten_scalars(self);'
        else:
            env['set_flags'] = SET_HISTORY.substitute(combined)
            if is_view:
                base_call = 'auto ret = as_view(static_cast<const Variable&>(self), {})'.format(base_call)
            else:
                base_call = 'auto ret = as_variable({})'.format(base_call)

        env['base_impl_call'] = base_call + ';'

        body.extend(METHOD_DEFINITION_BODY_DERIVATIVE.substitute(combined).split('\n'))
        return body

    def process_function(declaration):
        if skip_function(declaration):
            return

        env = {}

        strategy = dispatch_strategy(declaration)
        if strategy == 'use_derived':
            env['type_definition_body'] = emit_body_via_derived(declaration)
        elif strategy == 'use_type':
            env['type_definition_body'] = emit_body_via_type(declaration)
        else:
            # Hard failure here to encourage us to fix these methods
            # (rather than generate binding code which works for forward
            # but not backward).  In the limit, this case should never occur,
            # and we will replace this with an assert failure.
            env['type_definition_body'] = METHOD_DEFINITION_NYI.substitute(declaration)

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

        # Compute the Python function prototype for argument parsing
        typed_args = []
        positional = True
        for arg in declaration['arguments']:
            if arg.get('kwarg_only', False) and positional:
                typed_args.append('*')
                positional = False
            typename = arg['simple_type']
            if arg.get('is_nullable'):
                typename = '{}?'.format(typename)
            if arg.get('size') is not None:
                typename = '{}[{}]'.format(typename, arg['size'])
            param = typename + ' ' + arg['name']
            default = None
            if arg.get('default') is not None:
                default = arg['default']
                if default == 'nullptr' or default == '{}':
                    default = 'None'
            if arg.get('python_default_init') is not None:
                default = 'None'
            if default is not None:
                param += '=' + str(default)
            typed_args.append(param)

        # Python function prototype.
        # This is the string that we give to FunctionParameter, which is
        # then parsed into the actual structure which we do parsing
        # with.
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

    declarations_by_name = defaultdict(list)
    for d in aten_decls:
        declarations_by_name[d['name']].append(d)

    autograd_functions = load_derivatives(derivatives_path, declarations_by_signature, declarations_by_name)
    ensure_unique_names(autograd_functions)

    def should_generate_python_binding(declaration):
        name = declaration['name']
        # don't bind unimplemented functions to prevent errors in test_autograd.
        if dispatch_strategy(declaration) == 'unimplemented':
            return False

        for pattern in SKIP_PYTHON_BINDINGS:
            if re.match('^' + pattern + '$', name):
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

    create_autograd_functions(env, autograd_functions)
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
