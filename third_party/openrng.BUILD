cc_library(
    name = "openrng",
    srcs = glob([
        "openrng/*.c",
        "openrng/*.cpp"
    ], exclude = [
        "openrng/src/vsl/x86_64/*.c",
        "openrng/src/vsl/x86_64/*.cpp"
    ]),
    hdrs = glob([
        "openrng/*.h",
        "openrng/*.hpp"
    ]),
    includes = [
        "install/include/",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
)
