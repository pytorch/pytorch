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


OPTIONAL_TYPE_PATTERN = re.compile(r"c10::optional<(.+)>")
TYPE_PATTERN = re.compile(r"(?:const\s+)?([A-Z]\w+)")


def fully_qualified_type(argument_type):
    def maybe_optional_type(t, opt_match):
        return 'c10::optional<{}>'.format(t) if opt_match else t

    opt_match = OPTIONAL_TYPE_PATTERN.match(argument_type)
    if opt_match:
        argument_type = argument_type[opt_match.start(1):opt_match.end(1)]
    match = TYPE_PATTERN.match(argument_type)
    if match is None:
        return maybe_optional_type(argument_type, opt_match)
    index = match.start(1)
    qualified_type = "{}at::{}".format(argument_type[:index], argument_type[index:])
    return maybe_optional_type(qualified_type, opt_match)


def gen_variable_factories(out, declarations, template_path, disable_autograd=False, disable_trace=False):
    function_definitions = []
    for decl in declarations:
        has_tensor_options = any(a["simple_type"] == "TensorOptions" for a in decl["arguments"])
        is_namespace_fn = 'namespace' in decl['method_of']
        if (has_tensor_options or decl["name"].endswith("_like")) and is_namespace_fn:
            function_definitions.append(
                process_function(
                    decl,
                    has_tensor_options,
                    disable_autograd=disable_autograd,
                    disable_trace=disable_trace
                )
            )
    write(out,
          "variable_factories.h",
          CodeTemplate.from_file(template_path + "/variable_factories.h"),
          {"function_definitions": function_definitions})


def process_function(decl, has_tensor_options, disable_autograd, disable_trace):
    formals = []
    actuals = []
    for argument in decl["arguments"]:
        type = fully_qualified_type(argument["type"])
        default = " = {}".format(argument["default"]) if "default" in argument else ""
        formals.append("{} {}{}".format(type, argument["name"], default))
        actual = argument["name"]
        if argument["simple_type"] == "TensorOptions":
            actual = "at::TensorOptions({})".format(actual)
        actuals.append(actual)
    requires_grad = "options.requires_grad()" if has_tensor_options else "false"

    if not disable_autograd:
        pre_record_trace, post_record_trace = format_trace(decl, disable_trace)
    else:
        pre_record_trace, post_record_trace = '', ''

    return FUNCTION_TEMPLATE.substitute(
        name=decl["name"], formals=formals, actuals=actuals, requires_grad=requires_grad,
        pre_record_trace=pre_record_trace, post_record_trace=post_record_trace
    )
