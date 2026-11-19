load(
    "//xplat/third-party/XNNPACK/XNNPACK:build_srcs.bzl",
    _LOGGING_SRCS = "LOGGING_SRCS",
    _OPERATOR_SRCS = "OPERATOR_SRCS",
    _SUBGRAPH_SRCS = "SUBGRAPH_SRCS",
    _TABLE_SRCS = "TABLE_SRCS",
)
load("//xplat/third-party/XNNPACK/XNNPACK/gen:microkernels.bzl", "prod_srcs_for_arch")
load("//tools/build_defs:glob_defs.bzl", "subdir_glob")

_XNNPACK_SRCS = [
    "src/configs/argmaxpool-config.c",
    "src/configs/avgpool-config.c",
    "src/configs/binary-elementwise-config.c",
    "src/configs/cmul-config.c",
    "src/configs/conv-hwc2chw-config.c",
    "src/configs/dwconv-config.c",
    "src/configs/dwconv2d-chw-config.c",
    "src/configs/gemm-config.c",
    "src/configs/ibilinear-chw-config.c",
    "src/configs/ibilinear-config.c",
    "src/configs/lut32norm-config.c",
    "src/configs/maxpool-config.c",
    "src/configs/pack-lh-config.c",
    "src/configs/raddstoreexpminusmax-config.c",
    "src/configs/reduce-config.c",
    "src/configs/spmm-config.c",
    "src/configs/transpose-config.c",
    "src/configs/unary-elementwise-config.c",
    "src/configs/unpool-config.c",
    "src/configs/vmulcaddc-config.c",
    "src/configs/xx-fill-config.c",
    "src/configs/xx-pad-config.c",
    "src/configs/x8-lut-config.c",
    # Restored to keep the channel-shuffle operator (removed upstream) available
    # for ATen; see the restored src/x8-zip and src/x32-zip microkernels.
    "src/configs/zip-config.c",
]

def define_xnnpack_build_src(xnnpack_build_src):
    return ["XNNPACK/{}".format(src) for src in xnnpack_build_src]

def prod_srcs_for_arch_wrapper(arch):
    prod_srcs = prod_srcs_for_arch(arch)
    return define_xnnpack_build_src(prod_srcs)

def get_xnnpack_headers():
    src_headers = subdir_glob([
        ("XNNPACK", "src/**/*.h"),
        # Microkernel definitions are textually included (.inc) by the
        # microkernel sources, so they must be visible as headers.
        ("XNNPACK", "src/**/*.inc"),
    ])
    include_headers = subdir_glob([
        ("XNNPACK", "include/*.h"),
    ])

    return src_headers | include_headers

OPERATOR_SRCS = define_xnnpack_build_src(_OPERATOR_SRCS)
SUBGRAPH_SRCS = define_xnnpack_build_src(_SUBGRAPH_SRCS)
TABLE_SRCS = define_xnnpack_build_src(_TABLE_SRCS)
XNNPACK_SRCS = define_xnnpack_build_src(_XNNPACK_SRCS)
LOGGING_SRCS = define_xnnpack_build_src(_LOGGING_SRCS)
