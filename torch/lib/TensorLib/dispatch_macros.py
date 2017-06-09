from code_template import CodeTemplate

CASE_TEMPLATE = CodeTemplate("""\
case ${TypeID}:
    the_function<${specializations}>(the_type,__VA_ARGS__);
    break;
""")

MACRO_TEMPLATE = CodeTemplate("""\
#define ${macro_name}(the_type,the_function,...)
    switch(the_type.ID()) {
        ${cases}
    }
""")


def create_dispatch(all_types, include_type, include_processor):
    cases = []
    macro_name = "TLIB_DISPATCH"
    if include_type:
        macro_name += "_TYPE"
    if include_processor:
        macro_name += "_PROCESSOR"
    for typ in all_types:
        specializations = []
        if include_type:
            specializations.append(typ['ScalarType'])
        if include_processor:
            specializations.append('tlib::{}Tag'.format(typ['Processor']))
        cases.append(CASE_TEMPLATE.substitute(
            typ, specializations=specializations))
    the_macro = MACRO_TEMPLATE.substitute(macro_name=macro_name, cases=cases)
    # end lines in backslashes to make defines
    return '\\\n'.join(the_macro.split('\n')) + '\n'


def create(all_types):
    return "#pragma once\n\n" + (create_dispatch(all_types, True, False) +
                                 create_dispatch(all_types, False, True) +
                                 create_dispatch(all_types, True, True))
