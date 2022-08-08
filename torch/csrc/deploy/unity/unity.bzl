load("@fbcode_macros//build_defs:cpp_library.bzl", "cpp_library")
load("@fbcode_macros//build_defs:native_rules.bzl", "cxx_genrule")
load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")

# @lint-ignore-every BUCKLINT
load("@fbsource//tools/build_defs:fb_native_wrapper.bzl", "fb_native")

def build_unity(name, **kwargs):
    python_binary(name = name, **kwargs)

    cxx_genrule(
        name = "{}_build_python_app_lib".format(name),
        out = "python_app.a",
        cmd = """\
        cp $(location :""" + name + """) python_app
        ld -r -b binary -o ${TMP}/python_app.o python_app
        # rename the .data section to .torch_deploy_payload.unity.
        # don't set the alloc/load flags for the section so it will not join
        # the party of relocation.
        # Also strip the _binary_python_app_start/end/size symbols to avoid
        # confusion.
        objcopy --rename-section .data=.torch_deploy_payload.unity,readonly,contents -N  _binary_python_app_start -N  _binary_python_app_end -N  _binary_python_app_size ${TMP}/python_app.o
        ar rcs ${OUT} ${TMP}/python_app.o
        """,
    )

    fb_native.prebuilt_cxx_library(
        name = "{}_python_app_lib".format(name),
        visibility = ["PUBLIC"],
        link_whole = True,
        preferred_linkage = "static",
        static_lib = ":{}_build_python_app_lib".format(name),
    )

    cpp_library(
        name = "{}_unity_lib".format(name),
        srcs = [
        ],
        linker_flags = [
            "--export-dynamic",
        ],
        exported_deps = [
            "//caffe2/torch/csrc/deploy/unity:unity_core",
            ":{}_python_app_lib".format(name),
        ],
    )
