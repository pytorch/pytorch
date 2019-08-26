# Generates C++ functions that wrap ATen tensor factory methods to turn them into Variables.
#
# This writes one file: variable_factories.h

import re

from .utils import CodeTemplate, write
from .gen_variable_type import format_trace


FUNCTION_TEMPLATE = CodeTemplate("""\
inline at::Tensor ${name}(${formals}) {
  ${pre_record_trace}
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::${name}(${actuals});
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/${requires_grad});
  ${post_record_trace}
  return result;
}
""")


TYPE_PATTERN = re.compile(r"(?:const\s+)?([A-Z]\w+)")

def collapseActualsTO(actuals):
    if 'dtype' in actuals and \
       'layout' in actuals and \
       'device' in actuals and \
       'pin_memory' in actuals:
       index = actuals.index('dtype')
       if index != -1:
           actuals.pop(index + 3)
           actuals.pop(index + 2)
           actuals.pop(index + 1)
           actuals.pop(index)
           actuals.insert(index, 'at::TensorOptions(options).is_variable(false)')
    
    if 'options' in actuals:
        index = actuals.index('options')
        actuals[index] = 'options.is_variable(false)'
    return actuals

def fix_c10_optional(type):
    if type == "c10::optional<ScalarType>":
        return "c10::optional<at::ScalarType>"

    if type == "c10::optional<Layout>":
        return "c10::optional<at::Layout>"

    if type == "c10::optional<Device>":
        return "c10::optional<at::Device>"

    return type

def collapseFormalsTO(formals):
    if ('ScalarType' in f for f in formals) and \
       ('Layout' in f for f in formals) and \
       ('Device' in f for f in formals) and \
       ('bool' in f for f in formals):

        index = -1

        if index == -1:
            index = formals.index('c10::optional<at::ScalarType> dtype = c10::nullopt') if 'c10::optional<at::ScalarType> dtype = c10::nullopt' in formals else -1
            if index != -1:
                formals.insert(index + 4, 'const at::TensorOptions & options = {}')

        if index == -1:
            index = formals.index('c10::optional<at::ScalarType> dtype = at::kLong') if 'c10::optional<at::ScalarType> dtype = at::kLong' in formals else -1
            if index != -1:
                formals.insert(index + 4, 'const at::TensorOptions & options = at::kLong')

        if index == -1:
            index = formals.index('at::ScalarType dtype') if 'at::ScalarType dtype' in formals else -1
            if index != -1:
                formals.insert(index + 4, 'const at::TensorOptions & options')

        if index != -1:
            formals.pop(index + 3)
            formals.pop(index + 2)
            formals.pop(index + 1)
            formals.pop(index)
    return formals

def fully_qualified_type(argument_type):
    match = TYPE_PATTERN.match(argument_type)
    if match is None:
        return argument_type
    index = match.start(1)
    return "{}at::{}".format(argument_type[:index], argument_type[index:])

def gen_variable_factories(out, declarations, template_path, disable_autograd=False):
    function_definitions = []
    for decl in declarations:
        is_tensor_option = any(arg['type'] == 'const TensorOptions &' for arg in decl['arguments'])
        is_namespace_fn = 'namespace' in decl['method_of']
        if (is_tensor_option or decl["name"].endswith("_like")) and is_namespace_fn:
            function_definitions.append(
                process_function(decl, is_tensor_option, disable_autograd=disable_autograd))

    write(out,
          "variable_factories.h",
          CodeTemplate.from_file(template_path + "/variable_factories.h"),
          {"function_definitions": function_definitions})

def process_function(decl, is_tensor_option, disable_autograd):
    formals = []
    actuals = []
    for argument in decl["arguments"]:
        type = fully_qualified_type(argument["type"])
        type = fix_c10_optional(type)
        default = " = {}".format(argument["default"]) if "default" in argument else ""
        if (default != ""):
            if default == ' = long':
                default = " = at::kLong"

            if default == " = False":
                default = " = false"

        formals.append("{} {}{}".format(type, argument["name"], default))
        actual = argument["name"]
        actuals.append(actual)
    formals = collapseFormalsTO(formals)
    actuals = collapseActualsTO(actuals) # <-- can be removed?

    requires_grad = "options.requires_grad()" if is_tensor_option else "false"
    if decl['name'].endswith('_like') and not is_tensor_option:
        # it's a tensor
        actuals.append('{}.options().is_variable(false)'.format(actuals[0]))

    if not disable_autograd:
        pre_record_trace, post_record_trace = format_trace(decl)
    elif 'options' in actuals:
        actuals[actuals.index('options')] = 'at::TensorOptions(options).is_variable(false)'
    else:
        pre_record_trace, post_record_trace = '', ''

    return FUNCTION_TEMPLATE.substitute(
        name=decl["name"], formals=formals, actuals=actuals, requires_grad=requires_grad,
        pre_record_trace=pre_record_trace, post_record_trace=post_record_trace
    )
