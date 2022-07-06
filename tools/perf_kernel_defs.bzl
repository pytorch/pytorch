load("@fbcode_macros//build_defs:cpp_library.bzl", "cpp_library")

is_dbg_build = native.read_config("fbcode", "build_mode", "").find("dbg") != -1
is_sanitizer = native.read_config("fbcode", "sanitizer", "") != ""

def define_perf_kernels(
        prefix,
        levels_and_flags,
        compiler_common_flags = [],
        arch_compiler_common_flags = {},
        dependencies = [],
        arch_dependencies = [],
        external_deps = []):
    vectorize_flags = ([
        # "-Rpass=loop-vectorize", # Add vectorization information to output
        "-DENABLE_VECTORIZATION=1",
        "-fveclib=SVML",
    ] if not is_dbg_build and not is_sanitizer else [])

    compiler_specific_flags = {
        "clang": vectorize_flags,
        "gcc": [],
    }

    compiler_specific_flags["clang"] += ["-Wno-pass-failed"]

    common_srcs = native.glob(
        ["**/*.cc"],
        exclude = [
            "**/*_avx512.cc",
            "**/*_avx2.cc",
            "**/*_avx.cc",
        ],
    )

    cpp_headers = native.glob(
        ["**/*.h"],
    )

    kernel_targets = {}
    for arch, levels_and_flags in levels_and_flags.items():
        for level, flags in levels_and_flags:
            cpp_library(
                name = prefix + "perfkernels_" + level,
                srcs = native.glob(["**/*_" + level + ".cc"]),
                headers = cpp_headers,
                compiler_flags = compiler_common_flags + flags,
                arch_compiler_flags = arch_compiler_common_flags,
                compiler_specific_flags = compiler_specific_flags,
                exported_deps = dependencies,
                exported_arch_deps = arch_dependencies,
                exported_external_deps = external_deps,
            )
            kernel_targets.setdefault(arch, []).append(":" + prefix + "perfkernels_" + level)

    cpp_library(
        name = prefix + "perfkernels",
        srcs = common_srcs,
        headers = cpp_headers,
        compiler_flags = compiler_common_flags,
        arch_compiler_flags = arch_compiler_common_flags,
        compiler_specific_flags = compiler_specific_flags,
        link_whole = True,
        exported_arch_deps = kernel_targets.items() + arch_dependencies,
        exported_deps = dependencies,
    )
