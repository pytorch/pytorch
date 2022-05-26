# @lint-ignore-every BUCKRESTRICTEDSYNTAX

def define_targets(rules):
    rules.cc_library(
        name = "c10",
        deps = [
            "//c10/core:CPUAllocator",
            "//c10/core:ScalarType",
            "//c10/core:alignment",
            "//c10/core:alloc_cpu",
            "//c10/core:base",
            "//c10/macros",
            "//c10/mobile:CPUCachingAllocator",
            "//c10/mobile:CPUProfilingAllocator",
            "//c10/util:TypeCast",
            "//c10/util:base",
            "//c10/util:typeid",
        ] + rules.if_cuda(
            [
                "//c10/cuda",
                "//c10/cuda:Macros",
            ],
            [],
        ),
        visibility = ["//visibility:public"],
    )

EXPORTED_PREPROCESSOR_FLAGS = [
    # We do not use cmake, hence we will use the following macro to bypass
    # the cmake_macros.h file.
    "-DC10_USING_CUSTOM_GENERATED_MACROS",
    # We will use glog in c10.
    "-DC10_USE_GLOG",
    # We will use minimal glog to reduce binary size.
    "-DC10_USE_MINIMAL_GLOG",
    # No NUMA on mobile.  Disable it so it doesn't mess with host tests.
    "-DC10_DISABLE_NUMA",
    # Match the C10_MOBILE setting from pt_defs.bzl.
    "-DC10_MOBILE",
]

DEFAULT_EXPORTED_FLAGS = EXPORTED_PREPROCESSOR_FLAGS + [
    # C10 will extensively use exceptions so we will enable it.
    "-fexceptions",
    # for static string objects in c10.
    "-Wno-global-constructors",
]

MSVC_EXPORTED_FLAGS = EXPORTED_PREPROCESSOR_FLAGS + [
    "/EHsc",
]

def c10_buck_target(
        name,
        rule_name,
        glob,
        subdir_glob,
        backtraces_compiler_flags = [],
        **kwargs):
    rule_name(
        name = name,
        srcs = glob(
            ["**/*.cpp"],
            exclude = [
                "test/**/*.cpp",
                "benchmark/**/*.cpp",
                "cuda/**/*.cpp",
            ],
        ),
        header_namespace = "c10",
        exported_headers = subdir_glob(
            [
                ("", "**/*.h"),
            ],
            exclude = [
                "test/**/*.h",
                "benchmark/**/*.h",
                "cuda/**/*.h",
            ],
        ),
        exported_preprocessor_flags = DEFAULT_EXPORTED_FLAGS,
        compiler_flags = [
            "-Werror",
            # Due to the use of global constructors (like registry), C10 will
            # need to be built without global constructor set as error.
            "-Wno-global-constructors",
            "-DDISABLE_NAMEDTENSOR",
        ] + backtraces_compiler_flags,
        link_whole = True,
        preprocessor_flags = [
            # This does not have effect when we build static libs in fbcode,
            # but the flag is to keep consistency between cmake and buck.
            # Note: this flags should NOT be in propagated pp flags.
            "-DC10_BUILD_MAIN_LIB",
        ],
        visibility = ["PUBLIC"],
        **kwargs
    )
