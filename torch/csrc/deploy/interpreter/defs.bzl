load("@fbcode_macros//build_defs:cpp_binary.bzl", "cpp_binary")
load("@fbcode_macros//build_defs:cpp_library.bzl", "cpp_library")
load("@fbcode_macros//build_defs:native_rules.bzl", "cxx_genrule")

# @lint-ignore-every BUCKLINT
load("@fbsource//tools/build_defs:fb_native_wrapper.bzl", "fb_native")

def embedded_interpreter(name, suffix, legacy = False, exported_deps = [], exported_external_deps = []):
    final_name = name
    is_all = suffix == "all"
    is_cuda = suffix == "cuda" or is_all
    is_hip = suffix == "hip"
    platform_static_lib = []
    for platform in ["platform009", "platform010"]:
        name = platform + "_" + final_name
        so_name = name + ".so"
        cpp_binary(
            name = so_name,
            srcs = [
                "interpreter_impl.cpp",
            ] + (["import_find_sharedfuncptr.cpp"] if is_all else []),
            headers = [
                "Optional.hpp",
                "interpreter_impl.h",
            ],
            header_namespace = "torch/csrc/deploy",
            dlopen_enabled = True,
            linker_flags = ([
                # This ensures only the intended interface symbols are public/global
                # the rest are hidden, regardless of how they were compiled
                # (e.g. fvisibility=hidden is NOT important for the component
                # objs in this library, since we override here.)
                "--version-script=$(location :hide_symbols.script)",
            ] if not is_all else []),
            deps = [
                "fbsource//third-party/fmt:fmt",
            ] + ([
                ":builtin_registry_cuda",
                "//caffe2:torch_python_cuda_without_torch",
                "//deeplearning/trt/python:frozen_tensorrt",
            ] if is_cuda else ([
                ":builtin_registry_hip",
                "//caffe2:torch_python_hip_without_torch",
            ] if is_hip else [
                ":builtin_registry",
                "//caffe2:torch_python_without_torch",
            ])),
            external_deps =
                [
                    # needed for interpreter.cpp itself, it uses pybind currently
                    ("frozenpython", None, "python-frozen"),
                    ("frozenpython", None, "python"),
                ],
            fbcode_platform = platform,
        )

        # We build torch::deploy with two embedded binaries- one with only cpu py bindings,
        # the other with cpu+cuda py bindings.  This unfortunately wastes some binary size,
        # but at least at runtime only one of them is loaded.
        #
        # This is becuase of two reasons
        # (1) that applications such as predictor want to depend on torch::deploy in a
        # cuda-agnostic way, e.g. they don't choose yet, and a binary/app that depends
        # on predictor either chooses to include or not include a dep on cuda.
        #
        # (2) the way the embedded binary is created and loaded, it only exposes a small
        # set of interface symbols globally, for creating a new interpreter, and hides its
        # other symbols (esp. python ones) so they don't conflict with other interpreters.
        # This prevents dividing the cpu and cuda portions of bindings into _separate_ libs
        # and loading the cuda part additively.  Hence to achieve requirement (1) we bundle
        # two complete interpreter libs, one with and one without cuda.

        cp_cmd = "$(location //caffe2/torch/csrc/deploy:remove_dt_needed)" if suffix == "all" else "cp"

        build_name = "build_" + name
        if not legacy:
            cxx_genrule(
                name = build_name,
                out = "embedded_interpreter_" + suffix + ".a",
                cmd = """\
                """ + cp_cmd + """ $(location :""" + so_name + """) libtorch_deployinterpreter_internal_""" + suffix + """.so
                $(exe fbsource//third-party/binutils:ld) -r \\
                -m """ + select({
                    "ovr_config//cpu:arm64": "aarch64linux",
                    "ovr_config//cpu:x86_64": "elf_x86_64",
                }) + """ \\
                -b binary -o ${TMP}/embedded_interpreter_""" + suffix + """.o libtorch_deployinterpreter_internal_""" + suffix + """.so
                 $(exe fbsource//third-party/binutils:objcopy) --rename-section .data=.torch_deploy_payload.interpreter_""" + suffix + """,readonly,contents -N _binary_libtorch_deployinterpreter_""" + suffix + """_so_start -N _binary_libtorch_deployinterpreter_""" + suffix + """_so_end ${TMP}/embedded_interpreter_""" + suffix + """.o
                $(exe fbsource//third-party/binutils:ar) rcs ${OUT} ${TMP}/embedded_interpreter_""" + suffix + """.o
                """,
            )
        else:
            cxx_genrule(
                name = build_name,
                out = "embedded_interpreter_cuda_legacy.a",
                cmd = """\
                cp $(location :""" + so_name + """) libtorch_deployinterpreter_cuda.so
                $(exe fbsource//third-party/binutils:ld) -r \\
                -m """ + select({
                    "ovr_config//cpu:arm64": "aarch64linux",
                    "ovr_config//cpu:x86_64": "elf_x86_64",
                }) + """ \\
                -b binary -o ${TMP}/embedded_interpreter_cuda.o libtorch_deployinterpreter_cuda.so
                $(exe fbsource//third-party/binutils:ar) rcs ${OUT} ${TMP}/embedded_interpreter_cuda.o
                """,
            )
        platform_static_lib.append(["^" + platform, ":" + build_name])

    internal_name = final_name + "_internal"
    fb_native.prebuilt_cxx_library(
        preferred_linkage = "static",
        name = internal_name,
        visibility = ["PUBLIC"],
        link_whole = True,
        platform_static_lib = platform_static_lib,
    )

    # a thin wrapper around :embedded_interpreter_internal to add --export-dynamic
    # linker flags. The flag will be propagated to cpp_binary. We don't require
    # cpp_binary to explicitly enable --export-dynamic any more. New usecases usually
    # forgot to do so and caused interpreter not found crash.
    cpp_library(
        name = final_name,
        linker_flags = [
            "--export-dynamic",
        ],
        exported_deps = [
            ":" + internal_name,
        ] + exported_deps,
        exported_external_deps = exported_external_deps,
    )
