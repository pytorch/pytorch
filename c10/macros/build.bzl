def define_targets(
        rules,
        build_shared_libs,
        use_gflags,
        use_glog,
        use_msvc_static_runtime,
        use_numa):
    rules.cc_library(
        name = "macros",
        hdrs = [
            "Macros.h",
            # Despite the documentation in Macros.h, Export.h is included
            # directly by many downstream files. Thus, we declare it as a
            # public header in this file.
            "Export.h",
        ],
        srcs = [":cmake_macros"],
        visibility = ["//visibility:public"],
    )

    rules.cmake_configure_file(
        name = "cmake_macros",
        src = "cmake_macros.h.in",
        out = "cmake_macros.h",
        definitions = {
            "C10_BUILD_SHARED_LIBS": build_shared_libs,
            "C10_USE_GFLAGS": use_gflags,
            "C10_USE_GLOG": use_glog,
            "C10_USE_MSVC_STATIC_RUNTIME": use_msvc_static_runtime,
            "C10_USE_NUMA": use_numa,
        },
    )
