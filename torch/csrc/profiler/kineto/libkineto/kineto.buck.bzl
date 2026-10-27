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
            "src/ActivityProfilerController.cpp",
            "src/ActivityProfilerProxy.cpp",
            "src/CuptiActivityApi.cpp",
            "src/CuptiActivityProfiler.cpp",
            "src/Demangle.cpp",
            "src/init.cpp",
            "src/output_csv.cpp",
            "src/output_json.cpp",
        ],
        headers = subdir_glob(
            [
                ("include", "*.h"),
                ("src", "*.h"),
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
            "//third_party:fmt",
        ],
    )

    cxx_library(
        name = "libkineto_api",
        srcs = [
            "src/libkineto_api.cpp",
        ],
        headers = subdir_glob(
            [
                ("include", "*.h"),
                ("src", "*.h"),
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
            "//third_party:fmt",
        ],
    )

    cxx_library(
        name = "config_loader",
        srcs = [
            "src/ConfigLoader.cpp",
        ],
        headers = subdir_glob(
            [
                ("include", "ActivityType.h"),
                ("src", "*.h"),
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
            "src/AbstractConfig.cpp",
            "src/ActivityType.cpp",
            "src/Config.cpp",
        ],
        compiler_flags = KINETO_COMPILER_FLAGS,
        public_include_directories = [
            "include",
            "src",
        ],
        raw_headers = glob([
            "include/*.h",
            "src/*.h",
        ]),
        exported_deps = [
            ":logger",
            ":thread_util",
            "//third_party:fmt",
        ],
    )

    cxx_library(
        name = "logger",
        srcs = [
            "src/ILoggerObserver.cpp",
            "src/Logger.cpp",
        ],
        compiler_flags = KINETO_COMPILER_FLAGS,
        public_include_directories = [
            "include",
            "src",
        ],
        raw_headers = [
            "include/ILoggerObserver.h",
            "include/ThreadUtil.h",
            "src/Logger.h",
            "src/LoggerCollector.h",
        ],
        exported_deps = [
            ":thread_util",
            "//third_party:fmt",
        ],
    )

    cxx_library(
        name = "base_logger",
        srcs = [
            "src/GenericTraceActivity.cpp",
        ],
        public_include_directories = [
            "include",
            "src",
        ],
        raw_headers = glob([
            "include/*.h",
            "src/*.h",
            "src/*.tpp",
        ]),
        exported_deps = [
            ":thread_util",
        ],
    )

    cxx_library(
        name = "thread_util",
        srcs = [
            "src/ThreadUtil.cpp",
        ],
        compiler_flags = KINETO_COMPILER_FLAGS,
        exported_preprocessor_flags = [
            "-DKINETO_NAMESPACE=libkineto",
        ],
        public_include_directories = [
            "include",
        ],
        raw_headers = [
            "include/ThreadUtil.h",
        ],
        exported_deps = [
            "//third_party:fmt",
        ],
    )

    cxx_library(
        name = "libkineto_headers",
        exported_headers = native.glob([
            "include/*.h",
        ]),
        public_include_directories = [
            "include",
        ],
        visibility = ["PUBLIC"],
    )
