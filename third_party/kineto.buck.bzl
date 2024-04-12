load("//tools/build_defs:glob_defs.bzl", "subdir_glob")

# kineto code should be updated to not have to
# suppress these warnings.
KINETO_COMPILER_FLAGS = [
    "-fexceptions",
    "-Wno-deprecated-declarations",
    "-Wno-unused-function",
    "-Wno-unused-private-field",
]

def define_kineto():
    cxx_library(
        name = "libkineto",
        srcs = [
            "kineto/libkineto/src/ActivityProfilerController.cpp",
            "kineto/libkineto/src/ActivityProfilerProxy.cpp",
            "kineto/libkineto/src/CuptiActivityApi.cpp",
            "kineto/libkineto/src/CuptiActivityProfiler.cpp",
            "kineto/libkineto/src/CuptiRangeProfilerApi.cpp",
            "kineto/libkineto/src/Demangle.cpp",
            "kineto/libkineto/src/init.cpp",
            "kineto/libkineto/src/output_csv.cpp",
            "kineto/libkineto/src/output_json.cpp",
        ],
        headers = subdir_glob(
            [
                ("kineto/libkineto/include", "*.h"),
                ("kineto/libkineto/src", "*.h"),
            ],
        ),
        compiler_flags = KINETO_COMPILER_FLAGS,
        # @lint-ignore BUCKLINT
        link_whole = True,
        visibility = ["PUBLIC"],
        exported_deps = [
            ":base_logger",
            ":libkineto_api",
            ":thread_util",
            ":fmt",
        ],
    )

    cxx_library(
        name = "libkineto_api",
        srcs = [
            "kineto/libkineto/src/libkineto_api.cpp",
        ],
        headers = subdir_glob(
            [
                ("kineto/libkineto/include", "*.h"),
                ("kineto/libkineto/src", "*.h"),
            ],
        ),
        compiler_flags = KINETO_COMPILER_FLAGS,
        # @lint-ignore BUCKLINT
        link_whole = True,
        visibility = ["PUBLIC"],
        exported_deps = [
            ":base_logger",
            ":config_loader",
            ":thread_util",
            ":fmt",
        ],
    )

    cxx_library(
        name = "config_loader",
        srcs = [
            "kineto/libkineto/src/ConfigLoader.cpp",
        ],
        headers = subdir_glob(
            [
                ("kineto/libkineto/include", "ActivityType.h"),
                ("kineto/libkineto/src", "*.h"),
            ],
        ),
        compiler_flags = KINETO_COMPILER_FLAGS,
        exported_deps = [
            ":config",
            ":thread_util",
        ],
    )

    cxx_library(
        name = "config",
        srcs = [
            "kineto/libkineto/src/AbstractConfig.cpp",
            "kineto/libkineto/src/ActivityType.cpp",
            "kineto/libkineto/src/Config.cpp",
        ],
        compiler_flags = KINETO_COMPILER_FLAGS,
        public_include_directories = [
            "kineto/libkineto/include",
            "kineto/libkineto/src",
        ],
        # @lint-ignore BUCKRESTRICTEDSYNTAX
        raw_headers = glob([
            "kineto/libkineto/include/*.h",
            "kineto/libkineto/src/*.h",
        ]),
        exported_deps = [
            ":logger",
            ":thread_util",
            ":fmt",
        ],
    )

    cxx_library(
        name = "logger",
        srcs = [
            "kineto/libkineto/src/ILoggerObserver.cpp",
            "kineto/libkineto/src/Logger.cpp",
        ],
        compiler_flags = KINETO_COMPILER_FLAGS,
        public_include_directories = [
            "kineto/libkineto/include",
            "kineto/libkineto/src",
        ],
        raw_headers = [
            "kineto/libkineto/include/ILoggerObserver.h",
            "kineto/libkineto/include/ThreadUtil.h",
            "kineto/libkineto/src/Logger.h",
            "kineto/libkineto/src/LoggerCollector.h",
        ],
        exported_deps = [
            ":thread_util",
            ":fmt",
        ],
    )

    cxx_library(
        name = "base_logger",
        srcs = [
            "kineto/libkineto/src/GenericTraceActivity.cpp",
        ],
        public_include_directories = [
            "kineto/libkineto/include",
            "kineto/libkineto/src",
        ],
        # @lint-ignore BUCKRESTRICTEDSYNTAX
        raw_headers = glob([
            "kineto/libkineto/include/*.h",
            "kineto/libkineto/src/*.h",
            "kineto/libkineto/src/*.tpp",
        ]),
        exported_deps = [
            ":thread_util",
        ],
    )

    cxx_library(
        name = "thread_util",
        srcs = [
            "kineto/libkineto/src/ThreadUtil.cpp",
        ],
        compiler_flags = KINETO_COMPILER_FLAGS,
        exported_preprocessor_flags = [
            "-DKINETO_NAMESPACE=libkineto",
        ],
        public_include_directories = [
            "kineto/libkineto/include",
        ],
        raw_headers = [
            "kineto/libkineto/include/ThreadUtil.h",
        ],
        exported_deps = [
            ":fmt",
        ],
    )

    cxx_library(
        name = "libkineto_headers",
        exported_headers = native.glob([
            "kineto/libkineto/include/*.h",
        ]),
        public_include_directories = [
            "kineto/libkineto/include",
        ],
        visibility = ["PUBLIC"],
    )
