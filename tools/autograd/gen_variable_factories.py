# Generates C++ functions that wrap ATen tensor factory methods to turn them into Variables.
#
# This writes one file: variable_factories.h

import re

from .utils import CodeTemplate, write
from .gen_variable_type import format_trace

from tools.shared.module_loader import import_module
TOUtils = import_module('tensor_options_utils', 'aten/src/ATen/tensor_options_utils.py')

# This is a hack.
# Please see [Use only optional version of tensor options when getting them from TensorOptions object]
# In the tracking issue https://github.com/pytorch/pytorch/issues/30405
FUNCTION_TEMPLATE_ARANGE = CodeTemplate("""\
inline at::Tensor _${name}(${formals}) {
  ${pre_record_trace}
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    if (dtype.has_value()) {
      return at::_arange(${uncollapsed_actuals});
    }
    return at::_arange(${uncollapsed_actuals_nullptr});
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  ${post_record_trace}
  return result;
}

inline at::Tensor ${name}(${collapsed_formals}) {
  if (options.has_dtype()) {
    return torch::_$name(${options_calls});
  } else {
    return torch::_$name(${options_calls_nullptr});
  }
}
""")

FUNCTION_TEMPLATE_TENSOR_OPTIONS = CodeTemplate("""\
inline at::Tensor _${name}(${formals}) {
  ${pre_record_trace}
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_${name}(${uncollapsed_actuals});
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/requires_grad.value());
  ${post_record_trace}
  return result;
}

inline at::Tensor ${name}(${collapsed_formals}) {
    return torch::_$name(${options_calls});
}
""")

FUNCTION_TEMPLATE = CodeTemplate("""\
inline at::Tensor ${name}(${formals}) {
  ${pre_record_trace}
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_${name}(${actuals});
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/${requires_grad});
  ${post_record_trace}
  return result;
}
""")

TYPE_PATTERN = re.compile(r"(?:const\s+)?([A-Z]\w+)")

def fully_qualified_type(argument_type):
    match = TYPE_PATTERN.match(argument_type)
    if match is None:
        return argument_type
    index = match.start(1)
    return "{}at::{}".format(argument_type[:index], argument_type[index:])


def gen_variable_factories(out, declarations, template_path, disable_autograd=False):
    function_definitions = []
    for decl in declarations:
        has_tensor_options = TOUtils.check_if_factory_method(decl["arguments"])
        is_namespace_fn = 'namespace' in decl['method_of']
        if (has_tensor_options or decl["name"].endswith("_like")) and is_namespace_fn:
            function_definitions.append(
                process_function(decl, has_tensor_options, disable_autograd=disable_autograd))
    write(out,
          "variable_factories.h",
          CodeTemplate.from_file(template_path + "/variable_factories.h"),
          {"function_definitions": function_definitions})

def turn_actuals_into_option_calls(actuals):
    collapsed = actuals[:]
    index = actuals.index('dtype')
    collapsed[index] = 'at::typeMetaToScalarType(options.dtype())'
    collapsed[index + 1] = 'options.layout()'
    collapsed[index + 2] = 'options.device()'
    collapsed[index + 3] = 'options.pinned_memory()'
    collapsed.insert(index + 4, 'options.requires_grad()')
    #collapsed[index + 4] = 'options.requires_grad()'
    return collapsed

def process_function(decl, has_tensor_options, disable_autograd):
    formals = []
    actuals = []
    for argument in decl["arguments"]:
        type = fully_qualified_type(argument["type"])

        default = " = {}".format(argument["default"]) if "default" in argument else ""
        if "default" in argument:
            default = default.replace("False", "false")
            default = default.replace("True", "true")

        formals.append("{} {}{}".format(type, argument["name"], default))

        if argument['name'] == 'pin_memory' and has_tensor_options:
            #insert grad before MemoryFormat
            formals.append("c10::optional<bool> requires_grad = c10::nullopt")

        actual = argument["name"]
        if argument["simple_type"] == "TensorOptions":
            actual = "at::TensorOptions({})".format(actual)
        actuals.append(actual)

    if decl['name'].endswith('_like') and not has_tensor_options:
        # Insert TensorOptions before MemoryFormat
        actuals.insert(-1, 'at::typeMetaToScalarType({}.options().dtype())'.format(actuals[0]))
        actuals.insert(-1, '{}.options().layout()'.format(actuals[0]))
        actuals.insert(-1, '{}.options().device()'.format(actuals[0]))
        actuals.insert(-1, '{}.options().pinned_memory()'.format(actuals[0]))

    if not disable_autograd:
        pre_record_trace, post_record_trace = format_trace(decl)
    else:
        pre_record_trace, post_record_trace = '', ''

    requires_grad = "false"
    if not has_tensor_options:
        return FUNCTION_TEMPLATE.substitute(
            name=decl["name"], formals=formals, actuals=actuals, requires_grad=requires_grad,
            pre_record_trace=pre_record_trace, post_record_trace=post_record_trace
        )
    else:
        options_calls = turn_actuals_into_option_calls(actuals)
        collapsed_formals = TOUtils.collapse_formals(formals)


        if decl['name'] == 'arange':
            uncollapsed_actuals_nullptr = actuals[:]
            index = actuals.index('dtype')
            uncollapsed_actuals_nullptr[index] = 'c10::nullopt'

            options_calls_nullptr = options_calls[:]
            index = options_calls.index('at::typeMetaToScalarType(options.dtype())')
            options_calls_nullptr[index] = 'c10::nullopt'

            return FUNCTION_TEMPLATE_ARANGE.substitute(
                name=decl["name"],
                formals=formals,
                collapsed_formals=collapsed_formals,
                actuals=actuals,
                options_calls=options_calls,
                options_calls_nullptr=options_calls_nullptr,
                uncollapsed_actuals=actuals,
                uncollapsed_actuals_nullptr=uncollapsed_actuals_nullptr,
                pre_record_trace=pre_record_trace,
                post_record_trace=post_record_trace
            )
        else:
            return FUNCTION_TEMPLATE_TENSOR_OPTIONS.substitute(
                name=decl["name"],
                formals=formals,
                collapsed_formals=collapsed_formals,
                actuals=actuals,
                options_calls=options_calls,
                uncollapsed_actuals=actuals,
                pre_record_trace=pre_record_trace,
                post_record_trace=post_record_trace
            )
