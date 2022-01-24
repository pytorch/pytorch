def define_targets(rules):
    rules.cc_library(
        name = "ConstexprCrc",
        hdrs = ["ConstexprCrc.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":IdWrapper",
            ":min",
        ],
    )

    rules.cc_library(
        name = "ExclusivelyOwned",
        hdrs = ["ExclusivelyOwned.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [":min"],
    )

    rules.cc_library(
        name = "IdWrapper",
        hdrs = ["IdWrapper.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = ["//c10/macros"],
    )

    rules.cc_library(
        name = "MaybeOwned",
        hdrs = ["MaybeOwned.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":min",
            "//c10/macros",
        ],
    )

    rules.cc_library(
        name = "ThreadLocal",
        hdrs = ["ThreadLocal.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = ["//c10/macros"],
    )

    rules.cc_library(
        name = "ThreadLocalDebugInfo",
        srcs = ["ThreadLocalDebugInfo.cpp"],
        hdrs = ["ThreadLocalDebugInfo.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":ThreadLocal",
            ":min",
            "//c10/macros",
        ],
    )

    rules.cc_library(
        name = "TypeCast",
        srcs = ["TypeCast.cpp"],
        hdrs = ["TypeCast.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":base",
            "//c10/core:ScalarType",
            "//c10/macros",
        ],
    )

    rules.cc_library(
        name = "TypeIndex",
        hdrs = ["TypeIndex.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":ConstexprCrc",
            ":IdWrapper",
            ":min",
        ],
    )

    rules.cc_library(
        name = "UniqueVoidPtr",
        srcs = ["UniqueVoidPtr.cpp"],
        hdrs = ["UniqueVoidPtr.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = ["//c10/macros"],
    )

    rules.cc_library(
        name = "accumulate",
        hdrs = ["accumulate.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [":min"],
    )

    rules.cc_library(
        name = "base",
        srcs = [
            "DeadlockDetection.cpp",
            "LeftRight.cpp",
            "MathConstants.cpp",
            "SmallVector.cpp",
            "Unicode.cpp",
            "int128.cpp",
            "numa.cpp",
            "signal_handler.cpp",
            "thread_name.cpp",
        ],
        hdrs = [
            "BFloat16-math.h",
            "Bitset.h",
            "DeadlockDetection.h",
            "FunctionRef.h",
            "LeftRight.h",
            "MathConstants.h",
            "ScopeExit.h",
            "SmallBuffer.h",
            "Unicode.h",
            "Unroll.h",
            "copysign.h",
            "either.h",
            "env.h",
            "flat_hash_map.h",
            "hash.h",
            "int128.h",
            "numa.h",
            "order_preserving_flat_hash_map.h",
            "overloaded.h",
            "signal_handler.h",
            "sparse_bitset.h",
            "tempfile.h",
            "thread_name.h",
            "variant.h",
            "win32-headers.h",
        ],
        # This library uses flags and registration. Do not let the
        # linker remove them.
        alwayslink = True,
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":flat_hash_map",
            ":IdWrapper",
            ":llvmMathExtras",
            ":min",
            ":types",
            "@fmt",
            "//c10/macros",
        ],
    )

    rules.cc_library(
        name = "flat_hash_map",
        hdrs = ["flat_hash_map.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = ["//c10/macros"],
    )

    rules.cc_library(
        name = "intrusive_ptr",
        srcs = ["intrusive_ptr.cpp"],
        hdrs = ["intrusive_ptr.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":ExclusivelyOwned",
            ":MaybeOwned",
            ":min",
        ],
    )

    rules.cc_library(
        name = "llvmMathExtras",
        hdrs = ["llvmMathExtras.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
    )

    rules.cc_library(
        name = "min",
        srcs = [
            "Array.cpp",
            "Backtrace.cpp",
            "C++17.cpp",
            "Exception.cpp",
            "Logging.cpp", #SanitizeThread.h, android/log.h
            "Metaprogramming.cpp",
            "Optional.cpp",
            "SmallVector.cpp",
            "StringUtil.cpp",
            "TypeList.cpp",
            "TypeTraits.cpp",
            "Type_demangle.cpp",
            "Type_no_demangle.cpp",
            "flags_use_gflags.cpp",  # ought to really be in min
            "flags_use_no_gflags.cpp",  # ought to really be in min
        ],
        hdrs = [
            "AlignOf.h",
            "Array.h",
            "ArrayRef.h",
            "Backtrace.h",
            "C++17.h",
            "Deprecated.h",
            "Exception.h",
            "Flags.h",
            "Logging.h",
            "Metaprogramming.h",
            "Optional.h",
            "Registry.h",
            "SmallVector.h",
            "StringUtil.h",
            "Type.h",
            "TypeList.h",
            "TypeTraits.h",
            "in_place.h",
            "irange.h",
            "logging_is_google_glog.h",
            "logging_is_not_google_glog.h",
            "reverse_iterator.h",
            "string_utils.h",
            "string_view.h",
        ],
        deps = [
            "//c10/macros",
        ] + rules.select({
            "//c10:using_gflags": ["@com_github_gflags_gflags//:gflags"],
            "//conditions:default": [],
        }) + rules.select({
            "//c10:using_glog": ["@com_github_glog//:glog"],
            "//conditions:default": [],
        }),
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
    )

    rules.cc_library(
        name = "python_stub",
        hdrs = ["python_stub.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
    )

    rules.cc_library(
        name = "typeid",
        srcs = ["typeid.cpp"],
        hdrs = ["typeid.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":IdWrapper",
            ":TypeIndex",
            ":flat_hash_map",
            ":min",
            "//c10/core:ScalarType",
            "//c10/macros",
        ],
    )

    rules.cc_library(
        name = "types",
        srcs = [
            "Half.cpp",
            "complex_math.cpp",
        ],
        hdrs = [
            "BFloat16-inl.h",
            "BFloat16.h",
            "Half-inl.h",
            "Half.h",
            "TypeSafeSignMath.h",
            "complex.h",
            "complex_math.h",
            "complex_utils.h",
            "math_compat.h",
            "qint32.h",
            "qint8.h",
            "quint2x4.h",
            "quint4x2.h",
            "quint8.h",
        ],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":min",
            "//c10/macros",
        ],
    )

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(
            ["*.h"],
            exclude = [
            ],
        ),
        visibility = ["//c10:__pkg__"],
    )
