def define_rules(rules):
    rules.cc_library(
        name = "Export",
        hdrs = ["Export.h"],
        visibility = ["//:__subpackages__"],
    )

    rules.cc_library(
        name = "Macros",
        hdrs = ["Macros.h"],
        deps = [
            ":Export",
            ":cmake_macros",
        ],
        visibility = ["//:__subpackages__"],
    )

    rules.header_template(
        name = "cmake_macros",
        src = "cmake_macros.h.in",
        out = "cmake_macros.h",
        substitutions = {
            "cmakedefine": "define",
            "#define C10_USE_NUMA": "/* #undef C10_USE_NUMA */",
        },
    )
