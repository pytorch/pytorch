GLOG_SRCS = [
    'src/demangle.cc',
    'src/vlog_is_on.cc',
    'src/symbolize.cc',
    'src/raw_logging.cc',
    'src/logging.cc',
    'src/signalhandler.cc',
    'src/utilities.cc',
]

GLOG_CONFIG_HEADERS = [
    'vlog_is_on',
    'stl_logging',
    'raw_logging',
    'logging',
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

    native.cxx_library(
        name = "glog",
        srcs = [":glog_srcs[{}]".format(src) for src in GLOG_SRCS],
        # headers = [":glog_srcs[config.h]"],
        compiler_flags = [
            '-Wno-sign-compare',
            '-Wno-unused-function',
            '-Wno-unused-local-typedefs',
            '-Wno-unused-variable',
            '-Wno-deprecated-declarations',
        ],
        preferred_linkage = "static",
        exported_linker_flags = [],
        exported_preprocessor_flags = [
            '-DGLOG_NO_ABBREVIATED_SEVERITIES',
            '-DGLOG_STL_LOGGING_FOR_UNORDERED',
            '-DGOOGLE_GLOG_DLL_DECL=',
            '-DGOOGLE_NAMESPACE=google',
            # this is required for buck build
            '-DGLOG_BAZEL_BUILD',
            '-DHAVE_PTHREAD',
            # Allows src/logging.cc to determine the host name.
            '-DHAVE_SYS_UTSNAME_H',
            # For src/utilities.cc.
            '-DHAVE_SYS_SYSCALL_H',
            '-DHAVE_SYS_TIME_H',
            '-DHAVE_STDINT_H',
            '-DHAVE_STRING_H',
            # Enable dumping stacktrace upon sigaction.
            '-DHAVE_SIGACTION',
            # For logging.cc.
            '-DHAVE_PREAD',
            '-DHAVE___ATTRIBUTE__',
            '-I$(location :glog_headers)/../',
        ],
        deps = [":glog_http_archive"],
        soname = "libglog.$(ext)",
        visibility = ["PUBLIC"],
    )

    native.genrule(
        name = "glog_srcs",
        outs = {src: [src] for src in GLOG_SRCS},
        cmd = " && ".join([
            "rsync -a $(location :glog_http_archive)/ $OUT",
            "awk '{ gsub(/^#cmakedefine/, \"//cmakedefine\"); print; }' $(location :glog_http_archive)/src/config.h.cmake.in > $OUT/src/config.h",
            "awk '{ gsub(/^#cmakedefine/, \"//cmakedefine\"); print; }' $(location :glog_http_archive)/src/config.h.cmake.in > $OUT/src/base/config.h",
        ]),
        # reuqired for internal buck but not oss buck
        # default_outs = ["."],
    )

    native.genrule(
        name = "glog_headers",
        out = "glog",
        cmd = " && ".join([
            "rsync -a $(location :glog_http_archive)/src/glog/log_severity.h $OUT/",
        ] + [
            "{} $(location :glog_http_archive)/src/glog/{}.h.in > $OUT/{}.h".format(GLOG_SED_COMMAND, f, f) for f in GLOG_CONFIG_HEADERS
            ],
        ),
        # reuqired for internal buck but not oss buck
        # default_outs = ["."],
    )

    native.http_archive(
        name = "glog_http_archive",
        strip_prefix = "glog-0.4.0",
        sha256 = "f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c",
        urls = [
            "https://github.com/google/glog/archive/v0.4.0.tar.gz",
        ],
        out = "",
    )
