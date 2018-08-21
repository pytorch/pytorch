# Generates C++ functions that wrap ATen tensor factory methods to turn them into Variables.
#
# This writes one file: variable_factories.h

import re

from .utils import CodeTemplate, write

FUNCTION_TEMPLATE = CodeTemplate("""\
inline at::Tensor ${name}(${formals}) {
  at::Tensor tensor = at::${name}(${actuals});
  return autograd::make_variable(tensor, /*requires_grad=*/${requires_grad});
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
        if has_tensor_options or decl["name"].endswith("_like"):
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
            # We want to discard the runtime type so that `at::{name}` always returns a
            # tensor and not a variable, since we create a variable right after.
            actual += ".discard_runtime_type()"
        actuals.append(actual)
    requires_grad = "options.requires_grad()" if has_tensor_options else "false"
    if decl['name'].endswith('_like') and not has_tensor_options:
        actuals.append('at::TensorOptions({}, /*discard_runtime_type=*/true)'.format(actuals[0]))
    return FUNCTION_TEMPLATE.substitute(
        name=decl["name"], formals=formals, actuals=actuals, requires_grad=requires_grad
    )
