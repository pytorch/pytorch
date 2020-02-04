# This script generates BackendSelectRegister.cpp which is being used for dispatching purposes.
# We process only those factory functions that have 'backend_select' flag in its native_functions.yaml definition.

from code_template import CodeTemplate
import tensor_options_utils as TOUtils

FUNCTION_REGISTRATION = CodeTemplate("""\
.op(torch::RegisterOperators::options()
  .schema("${schema_string}")
  .impl_unboxedOnlyKernel<decltype(${function_name}), &${function_name}>(DispatchKey::BackendSelect)
  .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
""")

FUNCTION_DEFINITION = CodeTemplate("""\
Tensor ${function_name}(${method_formals}) {
  DispatchKey key = TensorOptions::computeDispatchKey(${dispatch_key_args});
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::${name}", "${overload_name}");
  return op.callUnboxedWithDispatchKey<${formals_types}>(key, ${type_method_actuals});
}
""")

def register_backend_select_methods(declarations, template_path, file_manager):
    backend_select_method_definitions = []
    backend_select_function_registrations = []

    for decl in declarations:
        for option in decl["options"]:
            if not option.get('backend_select', False):
                continue

            name = option['name']
            if 'overload_name' in option and option['overload_name'] != '':
                name = "{0}_{1}".format(name, option['overload_name'])

            if "arguments" in option and TOUtils.check_if_factory_method(option["arguments"]):
                func_reg = FUNCTION_REGISTRATION.substitute(schema_string=option['schema_string'],
                                                            function_name=name)

                if TOUtils.check_hack(name):
                    dispatch_key_args = "dtype, layout, device"
                else:
                    dispatch_key_args = "dtype.value_or(ScalarType::Float), layout.value_or(kStrided), device.value_or(kCPU)"
                method_def = FUNCTION_DEFINITION.substitute(function_name=name,
                                                            method_formals=option['method_formals'],
                                                            name=option['name'],
                                                            dispatch_key_args=dispatch_key_args,
                                                            overload_name=option['overload_name'],
                                                            formals_types=option['formals_types_with_return'],
                                                            type_method_actuals=option['type_method_actuals'])

                backend_select_function_registrations.append(func_reg)
                backend_select_method_definitions.append(method_def)

    env = {}
    env['backend_select_method_definitions'] = backend_select_method_definitions
    env['backend_select_function_registrations'] = backend_select_function_registrations
    file_manager.write('BackendSelectRegister.cpp', template_path, env)
