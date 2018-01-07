# Generates C++ autograd functions for the derivatives of ATen operations
#
# This writes two files:
#  Functions.h/cpp: subclasses of autograd::Function
#  python_functions.h/cpp: Python bindings for the above classes
#
from .utils import nested_dict, CodeTemplate, write
from .gen_variable_type import VIEW_FUNCTIONS, uses_single_grad, template_path

FUNCTIONS_H = CodeTemplate.from_file(template_path + '/Functions.h')
FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/Functions.cpp')
PY_FUNCTIONS_H = CodeTemplate.from_file(template_path + '/python_functions.h')
PY_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_functions.cpp')

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

GRAD_INPUT_MASK = CodeTemplate("""\
  auto grad_input_mask = std::array<bool, ${n}>{
    ${masks}
  };\
""")

DERIVATIVE_MULTI = CodeTemplate("""\
if (should_compute_output({ ${idxs} })) {
${grad_input_mask}
  std::tie(${grad_inputs}) = ${derivative};
}
""")

DERIVATIVE_TENSORLIST = CodeTemplate("""\
if (should_compute_any_outputs()) {
  grad_inputs = ${derivative};
}
""")

# These functions have backwards which cannot be traced, and so must have
# their backward functions traced opaquely.
# VIEW_FUNCTIONS are not traceable because they use as_strided, which
# has an untraceable backwards, see
# https://github.com/pytorch/pytorch/issues/4250
# TODO: This is probably not exhaustive, but it's a start
UNTRACEABLE_FUNCTIONS = VIEW_FUNCTIONS


def gen_autograd_functions(out, autograd_functions):
    """Functions.h and Functions.cpp body

    These contain the auto-generated subclasses of torch::autograd::Function
    for each every differentiable torch function.
    """
    function_definitions = []
    function_declarations = []
    py_function_initializers = []

    for func in autograd_functions:
        env = process_function(func)

        function_declarations.append(FUNCTION_DECLARATION.substitute(env))
        function_definitions.append(FUNCTION_DEFINITION.substitute(env))
        py_function_initializers.append(PY_FUNCTION_DEFINITION.substitute(env))

    top_env = {
        'autograd_function_definitions': function_definitions,
        'autograd_function_declarations': function_declarations,
        'py_function_initializers': py_function_initializers,
    }

    write(out, 'Functions.h', FUNCTIONS_H, top_env)
    write(out, 'Functions.cpp', FUNCTIONS_CPP, top_env)
    write(out, 'python_functions.h', PY_FUNCTIONS_H, top_env)
    write(out, 'python_functions.cpp', PY_FUNCTIONS_CPP, top_env)


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

    if uses_single_grad(func):
        body.append('auto& grad = grads[0];')

    def emit_derivative(derivative):
        formula = derivative['formula']
        idxs = derivative['output_indices']
        if idxs == ['*']:
            return DERIVATIVE_TENSORLIST.substitute(derivative=formula)
        elif len(idxs) == 1:
            return DERIVATIVE_TENSOR.substitute(idx=idxs[0], derivative=formula)
        else:
            if 'grad_input_mask' in formula:
                masks = ['should_compute_output({}),'.format(i) for i in idxs]
                grad_input_mask = GRAD_INPUT_MASK.substitute(masks=masks, n=len(idxs))
            else:
                grad_input_mask = ''
            grad_inputs = ', '.join(['grad_inputs[{}]'.format(i) for i in idxs])
            return DERIVATIVE_MULTI.substitute(
                idxs=idxs, derivative=formula, grad_inputs=grad_inputs,
                grad_input_mask=grad_input_mask)

    body.extend(unpack)
    for derivative in func['derivatives']:
        body.append(emit_derivative(derivative))

    env['body'] = body
    if func['name'] in UNTRACEABLE_FUNCTIONS:
        env['superclass'] = 'Function'
    else:
        env['superclass'] = 'TraceableFunction'
    return nested_dict(env, func)
