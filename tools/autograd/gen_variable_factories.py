# Generates C++ functions that wrap ATen tensor factory methods to turn them into Variables.
#
# This writes one file: variable_factories.h

import re

from .utils import CodeTemplate, write


FUNCTION_TEMPLATE = CodeTemplate("""\
inline at::Tensor ${name}(${formals}) {
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::${name}(${actuals});
  })();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/${requires_grad});
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


def gen_variable_factories(out, declarations, template_path):
    function_definitions = []
    for decl in declarations:
        has_tensor_options = any(a["simple_type"] == "TensorOptions" for a in decl["arguments"])
        is_namespace_fn = 'namespace' in decl['method_of']
        if (has_tensor_options or decl["name"].endswith("_like")) and is_namespace_fn:
            function_definitions.append(
                process_function(
                    decl,
                    has_tensor_options,
                )
            )
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
            # note: we remove the requires_grad setting from the TensorOptions because
            # it is ignored anyways (and we actually have an assertion that it isn't set
            # which would fail otherwise). We handle requires_grad explicitly here
            # instead of passing it through to the kernel.
            actual = "at::TensorOptions({}).requires_grad(c10::nullopt)".format(actual)
        actuals.append(actual)
    requires_grad = "options.requires_grad()" if has_tensor_options else "false"

    return FUNCTION_TEMPLATE.substitute(
        name=decl["name"], formals=formals, actuals=actuals, requires_grad=requires_grad
    )
