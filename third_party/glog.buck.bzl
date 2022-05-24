GLOG_CONFIG_HEADERS = [
    "vlog_is_on.h",
    "stl_logging.h",
    "raw_logging.h",
    "logging.h",
]

GLOG_SED_COMMAND = " ".join([
    "sed",
    "-e 's/@ac_cv_cxx_using_operator@/1/g'",
    "-e 's/@ac_cv_have_unistd_h@/1/g'",
    "-e 's/@ac_cv_have_stdint_h@/1/g'",
    "-e 's/@ac_cv_have_systypes_h@/1/g'",
    "-e 's/@ac_cv_have_libgflags@/0/g'",
    "-e 's/@ac_cv_have_uint16_t@/1/g'",
    "-e 's/@ac_cv_have___builtin_expect@/1/g'",
    "-e 's/@ac_cv_have_.*@/0/g'",
    "-e 's/@ac_google_start_namespace@/namespace google {/g'",
    "-e 's/@ac_google_end_namespace@/}/g'",
    "-e 's/@ac_google_namespace@/google/g'",
    "-e 's/@ac_cv___attribute___noinline@/__attribute__((noinline))/g'",
    "-e 's/@ac_cv___attribute___noreturn@/__attribute__((noreturn))/g'",
    "-e 's/@ac_cv___attribute___printf_4_5@/__attribute__((__format__ (__printf__, 4, 5)))/g'",
])

def define_glog():
    cxx_library(
        name = "glog",
        srcs = [
            "glog/src/demangle.cc",
            "glog/src/vlog_is_on.cc",
            "glog/src/symbolize.cc",
            "glog/src/raw_logging.cc",
            "glog/src/logging.cc",
            "glog/src/signalhandler.cc",
            "glog/src/utilities.cc",
        ],
        exported_headers = [":glog_{}".format(header) for header in GLOG_CONFIG_HEADERS],
        header_namespace = "glog",
        compiler_flags = [
            "-Wno-sign-compare",
            "-Wno-unused-function",
            "-Wno-unused-local-typedefs",
            "-Wno-unused-variable",
            "-Wno-deprecated-declarations",
        ],
        preferred_linkage = "static",
        exported_linker_flags = [],
        exported_preprocessor_flags = [
            "-DGLOG_NO_ABBREVIATED_SEVERITIES",
            "-DGLOG_STL_LOGGING_FOR_UNORDERED",
            "-DGOOGLE_GLOG_DLL_DECL=",
            "-DGOOGLE_NAMESPACE=google",
            # this is required for buck build
            "-DGLOG_BAZEL_BUILD",
            "-DHAVE_PTHREAD",
            # Allows src/logging.cc to determine the host name.
            "-DHAVE_SYS_UTSNAME_H",
            # For src/utilities.cc.
            "-DHAVE_SYS_SYSCALL_H",
            "-DHAVE_SYS_TIME_H",
            "-DHAVE_STDINT_H",
            "-DHAVE_STRING_H",
            # Enable dumping stacktrace upon sigaction.
            "-DHAVE_SIGACTION",
            # For logging.cc.
            "-DHAVE_PREAD",
            "-DHAVE___ATTRIBUTE__",
        ],
        deps = [":glog_config"],
        soname = "libglog.$(ext)",
        visibility = ["PUBLIC"],
    )

    cxx_library(
        name = "glog_config",
        header_namespace = "",
        exported_headers = {
            "config.h": ":glog_config.h",
            "glog/log_severity.h": "glog/src/glog/log_severity.h",
        },
    )

    genrule(
        name = "glog_config.h",
        srcs = ["glog/src/config.h.cmake.in"],
        out = "config.h",
        cmd = "awk '{ gsub(/^#cmakedefine/, \"//cmakedefine\"); print; }' $SRCS > $OUT",
    )

    for header in GLOG_CONFIG_HEADERS:
        genrule(
            name = "glog_{}".format(header),
            out = header,
            srcs = ["glog/src/glog/{}.in".format(header)],
            cmd = "{} $SRCS > $OUT".format(GLOG_SED_COMMAND),
        )
