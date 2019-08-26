# Generates C++ functions that wrap ATen tensor factory methods to turn them into Variables.
#
# This writes one file: variable_factories.h

import re
import copy
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


def fix_c10_optional(type):
    if type == "c10::optional<ScalarType>":
        return "c10::optional<at::ScalarType>"

    if type == "c10::optional<Layout>":
        return "c10::optional<at::Layout>"

    if type == "c10::optional<Device>":
        return "c10::optional<at::Device>"

    return type

def fully_qualified_type(argument_type):
    match = TYPE_PATTERN.match(argument_type)
    if match is None:
        return argument_type
    index = match.start(1)
    return "{}at::{}".format(argument_type[:index], argument_type[index:])


def gen_variable_factories(out, declarations, template_path, disable_autograd=False):
    function_definitions = []
    for decl in declarations:
        a = any(arg['type'] == 'c10::optional<ScalarType>' for arg in decl['arguments']) and any(arg['type'] == 'c10::optional<Layout>' for arg in decl['arguments']) and any(arg['type'] == 'c10::optional<Device>' for arg in decl['arguments']) and any(arg['type'] == 'c10::optional<bool>' for arg in decl['arguments'])
        a1 = any(arg['type'] == 'c10::optional<at::ScalarType>' for arg in decl['arguments']) and any(arg['type'] == 'c10::optional<at::Layout>' for arg in decl['arguments']) and any(arg['type'] == 'c10::optional<at::Device>' for arg in decl['arguments']) and any(arg['type'] == 'c10::optional<bool>' for arg in decl['arguments'])
        b = any(arg['type'] == 'ScalarType' for arg in decl['arguments']) and any(arg['type'] == 'Layout' for arg in decl['arguments']) and any(arg['type'] == 'Device' for arg in decl['arguments']) and any(arg['type'] == 'bool' for arg in decl['arguments'])
        b1 = any(arg['type'] == 'at::ScalarType' for arg in decl['arguments']) and any(arg['type'] == 'at::Layout' for arg in decl['arguments']) and any(arg['type'] == 'at::Device' for arg in decl['arguments']) and any(arg['type'] == 'bool' for arg in decl['arguments'])
        c1 = any(arg['type'] == 'const TensorOptions &' for arg in decl['arguments'])
        is_tensor_option = a or b or a1 or b1 or c1

        is_namespace_fn = 'namespace' in decl['method_of']
        if (is_tensor_option or decl["name"].endswith("_like")) and is_namespace_fn:
            function_definitions.append(
                process_function(decl, is_tensor_option, disable_autograd=disable_autograd))
    write(out,
          "variable_factories.h",
          CodeTemplate.from_file(template_path + "/variable_factories.h"),
          {"function_definitions": function_definitions})

supported_topt_arguments = [
    [
        {'name': 'dtype', 'type': 'ScalarType', 'is_nullable': False, 'annotation': None},
        {'name': 'layout', 'type': 'Layout', 'is_nullable': False, 'annotation': None},
        {'name': 'device', 'type': 'Device', 'is_nullable': False, 'annotation': None},
        {'name': 'pin_memory', 'type': 'bool', 'is_nullable': False, 'annotation': None, 'default': False},
    ]
]
supported_topt_arguments.append(copy.deepcopy(supported_topt_arguments[0]))
for arg in supported_topt_arguments[1]:
    arg.update({'kwarg_only': True})
supported_topt_arguments.append(copy.deepcopy(supported_topt_arguments[1]))
for arg in supported_topt_arguments[2]:
    arg.update({'default': 'c10::nullopt', 'is_nullable': True})
# add explicit support for what is needed for tril_indices / triu_indices
supported_topt_arguments.append(
    [
        {'name': 'dtype', 'type': 'ScalarType', 'annotation': None, 'kwarg_only': True,
         'default': 'long', 'is_nullable': True},
        {'name': 'layout', 'type': 'Layout', 'annotation': None, 'kwarg_only': True,
         'default': 'c10::nullopt', 'is_nullable': True},
        {'name': 'device', 'type': 'Device', 'annotation': None, 'kwarg_only': True,
         'default': 'c10::nullopt', 'is_nullable': True},
        {'name': 'pin_memory', 'type': 'bool', 'annotation': None, 'kwarg_only': True,
         'default': 'c10::nullopt', 'is_nullable': True},
    ]
)

def is_tensor_option(argument):
    return argument['name'] in ['dtype', 'layout', 'device', 'pin_memory']

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

def process_function(decl, is_tensor_option, disable_autograd):
    foo = decl["name"].endswith("_like")
    if foo:
        print("\n\n", decl["name"])
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

    foo = decl['name'] == '_cudnn_init_dropout_state'
    if foo:
        print("\n\n\n\nNOW: ", actuals)
        print('isTO: ', is_tensor_option)
    actuals = collapseActualsTO(actuals) # <-- can be removed?

    if foo:
        print("NOW2: ", actuals)

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
