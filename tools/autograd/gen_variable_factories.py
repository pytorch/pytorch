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


def fully_qualified_type(argument_type):
    match = TYPE_PATTERN.match(argument_type)
    if match is None:
        return argument_type
    index = match.start(1)
    return "{}at::{}".format(argument_type[:index], argument_type[index:])


def gen_variable_factories(out, declarations, template_path):
    function_definitions = []
    for decl in declarations:
        has_tensor_options = any(a["simple_type"] == "TensorOptions" for a in decl["arguments"])
        is_namespace_fn = 'namespace' in decl['method_of']
        if (has_tensor_options or decl["name"].endswith("_like")) and is_namespace_fn:
            function_definitions.append(process_function(decl, has_tensor_options))
    write(out,
          "variable_factories.h",
          CodeTemplate.from_file(template_path + "/variable_factories.h"),
          {"function_definitions": function_definitions})


def process_function(decl, has_tensor_options):
    formals = []
    actuals = []
    for argument in decl["arguments"]:
        type = fully_qualified_type(argument["type"])
        default = " = {}".format(argument["default"]) if "default" in argument else ""
        formals.append("{} {}{}".format(type, argument["name"], default))
        actual = argument["name"]
        if argument["simple_type"] == "TensorOptions":
            # We want to make `at::{name}` always return a
            # tensor and not a variable, since we create a variable right after.
            actual = "at::TensorOptions({}).is_variable(false)".format(actual)
        actuals.append(actual)
    requires_grad = "options.requires_grad()" if has_tensor_options else "false"
    if decl['name'].endswith('_like') and not has_tensor_options:
        # it's a tensor
        actuals.append('{}.options().is_variable(false)'.format(actuals[0]))

    pre_record_trace, post_record_trace = format_trace(decl)

    return FUNCTION_TEMPLATE.substitute(
        name=decl["name"], formals=formals, actuals=actuals, requires_grad=requires_grad,
        pre_record_trace=pre_record_trace, post_record_trace=post_record_trace
    )
