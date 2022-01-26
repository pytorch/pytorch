def define_targets(rules):
    rules.cc_library(
        name = "ScalarType",
        hdrs = ["ScalarType.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = ["//c10/util:base"],
    )

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(
            [
                "*.h",
                "impl/*.h",
            ],
            exclude = [
            ],
        ),
        visibility = ["//c10:__pkg__"],
    )

    rules.filegroup(
        name = "sources",
        srcs = rules.glob(
            [
                "*.cpp",
                "impl/*.cpp",
            ],
            exclude = [
            ],
        ),
        visibility = ["//c10:__pkg__"],
    )
