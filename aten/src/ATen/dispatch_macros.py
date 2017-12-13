from code_template import CodeTemplate

BACKEND_CHECK_TEMPLATE = CodeTemplate("""\
if (the_type.backend() != Backend::${Backend}) {
    runtime_error("%s not implemented for '%s'", name, the_type.toString());
}
""")

CASE_TEMPLATE = CodeTemplate("""\
case ScalarType::${ScalarName}:
    return F<${ScalarType}>::apply(std::forward<Args>(args)...);
""")

MACRO_TEMPLATE = CodeTemplate("""\
template<template <typename> class F, typename ... Args>
auto dispatch_${dispatch_name}(const Type& the_type, const char *name, Args&&... args)
  -> decltype(F<double>::apply(std::forward<Args>(args)...)) {
    ${backend_check}
    switch(the_type.scalarType()) {
        ${cases}
        default:
            runtime_error("%s not implemented for '%s'", name, the_type.toString());
    }
}
""")


def create_dispatch(types, dispatch_name, backend_only):
    cases = []
    for typ in types:
        # Half CPU doesn't have math currently.
        if typ['Density'] != 'Sparse' and (backend_only != 'CPU' or typ['ScalarName'] != 'Half'):
            cases.append(CASE_TEMPLATE.substitute(typ))
    backend_check = [BACKEND_CHECK_TEMPLATE.substitute(typ)] if backend_only else []
    return MACRO_TEMPLATE.substitute(cases=cases,
                                     dispatch_name=dispatch_name,
                                     backend_check=backend_check)


def create(all_types):
    # since this is dispatching on scalar type, we can just select a backend with all scalar types
    # and not do a backend check
    all_cpu_types = [t for t in all_types if t['Backend'] == 'CPU']
    cpu_float_types = [t for t in all_types if t['Backend'] == 'CPU' and t['isFloatingType']]
    return create_dispatch(all_cpu_types, "all", False) + create_dispatch(cpu_float_types, "cpu_floating_types", 'CPU')
