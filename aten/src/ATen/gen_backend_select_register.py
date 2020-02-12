# This script generates BackendSelectRegister.cpp which is being used for dispatching purposes.
# We process only those factory functions that have 'backend_select' flag in its native_functions.yaml definition.

from code_template import CodeTemplate

GENERATED_COMMENT = CodeTemplate(
    "@" + "generated from ${filename}")

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
  return op.callUnboxedWithDispatchKey<${formals_types}>(key, ${type_method_actuals});
}
""")

def needs_backend_select(declaration_option):
    # We register an op under the BackendSelect dispatch key
    # if a TensorOptions argument has been gathered from its declared args
    # We skip all the 'new_*' and '*_like' ops as they are special cased and avoid dispatching.
    # See TypeDefault.cpp
    if declaration_option['name'].endswith('_like') or declaration_option['name'].startswith('new_'):
        return False

    return any(a.get('dynamic_type') == 'TensorOptions' for a in declaration_option['arguments'])

def register_backend_select_methods(declarations, template_path, file_manager):
    backend_select_method_definitions = []
    backend_select_function_registrations = []

    for decl in declarations:
        for option in decl["options"]:
            if needs_backend_select(option):
                assert option['use_c10_dispatcher'] == 'unboxed_only'

                name = option['name']
                if option.get('overload_name', '') != '':
                    name = "{0}_{1}".format(name, option['overload_name'])

                func_reg = FUNCTION_REGISTRATION.substitute(schema_string=option['schema_string'],
                                                            function_name=name)

                method_def = FUNCTION_DEFINITION.substitute(function_name=name,
                                                            method_formals=option['formals_with_defaults'],
                                                            name=option['name'],
                                                            overload_name=option['overload_name'],
                                                            formals_types=option['formals_types_with_return'],
                                                            type_method_actuals=option['type_method_actuals'])

                backend_select_function_registrations.append(func_reg)
                backend_select_method_definitions.append(method_def)

    env = {}
    env['backend_select_method_definitions'] = backend_select_method_definitions
    env['backend_select_function_registrations'] = backend_select_function_registrations

    env['generated_comment'] = GENERATED_COMMENT.substitute(filename=template_path)
    file_manager.write('BackendSelectRegister.cpp', template_path, env)
