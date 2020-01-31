import sys
import re
from code_template import CodeTemplate

FUNCTION_REGISTRATION = CodeTemplate("""\
.op(torch::RegisterOperators::options()
  .schema("${schema_string}")
  .impl_unboxedOnlyKernel<decltype(${function_name}), &${function_name}>(DispatchKey::BackendSelect)
  .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
""")

FUNCTION_DEFINITION = CodeTemplate("""\
Tensor ${function_name}(${method_formals}) {
  DispatchKey key = options.computeDispatchKey();
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::${name}", "${overload_name}");
  return op.callUnboxedWithDispatchKey<${formals_types_with_return}>(key, ${type_method_actuals});
}
""")

def register_backend_select_methods(declarations, top_env):
    backend_select_method_definitions = []
    backend_select_function_registrations = []

    for decl in declarations:
        for option in decl["options"]:
            if '_like' in option['name'] or 'new_' in option['name'] or 'to' in option['name']:
                continue

            name = option['name']
            if 'overload_name' in option and option['overload_name'] != '':
                name = "{0}_{1}".format(name, option['overload_name'])

            if "arguments" in option and any("dynamic_type" in a and a["dynamic_type"] == "TensorOptions" for a in option["arguments"]):
                backend_select_function_registrations.append(FUNCTION_REGISTRATION.substitute(schema_string=option['schema_string'],
                                                                                              function_name=name))
                backend_select_method_definitions.append(FUNCTION_DEFINITION.substitute(function_name=name, method_formals=option['method_formals'],
                                                                                        name=option['name'], overload_name=option['overload_name'],
                                                                                        formals_types_with_return=option['formals_types_with_return'],
                                                                                        type_method_actuals=option['type_method_actuals']))


    top_env['backend_select_method_definitions'] = backend_select_method_definitions
    top_env['backend_select_function_registrations'] = backend_select_function_registrations
