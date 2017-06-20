from code_template import CodeTemplate

CASE_TEMPLATE = CodeTemplate("""\
case ${TypeID}:
    return F<${ScalarType}>::${Backend}(the_type,std::forward<Args>(args)...);
""")

MACRO_TEMPLATE = CodeTemplate("""\
#pragma once

namespace at {

template<template <typename> class F, typename ... Args>
auto dispatch(const Type & the_type, Args&&... args)
    -> decltype(F<double>::CPU(the_type,std::forward<Args>(args)...)) {
    switch(the_type.ID()) {
        ${cases}
    }
}

}
""")


def create_dispatch(all_types):
    cases = []
    for typ in all_types:
        cases.append(CASE_TEMPLATE.substitute(typ))
    return MACRO_TEMPLATE.substitute(cases=cases)


def create(all_types):
    return create_dispatch(all_types)
