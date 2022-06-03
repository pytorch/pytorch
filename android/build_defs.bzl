load("@fbsource//tools/build_defs:fb_xplat_cxx_test.bzl", "fb_xplat_cxx_test")
load("@fbsource//xplat/caffe2:pt_defs.bzl", "get_build_from_deps_query", "pt_operator_registry")

DEFAULT_PT_OP_DEPS = [
    "fbsource//xplat/caffe2:torch_mobile_ops_full_dev",
]

def pt_xplat_cxx_test(name, deps = [], pt_op_deps = DEFAULT_PT_OP_DEPS, **kwargs):
    code_gen_lib = []
    if get_build_from_deps_query():
        lib_name = name + "_lib"
        pt_operator_registry(lib_name, preferred_linkage = "static", template_select = False, deps = pt_op_deps)
        code_gen_lib = [":" + lib_name]
        deps = deps + code_gen_lib
    fb_xplat_cxx_test(
        name = name,
        deps = deps,
        **kwargs
    )
