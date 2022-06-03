# @nolint
load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library")
load("@fbsource//xplat/caffe2/c10:ovrsource_defs.bzl", "cpu_supported_platforms")

def define_caffe2_perfkernels():
    [
        oxx_static_library(
            name = "perfkernels_{}_ovrsource".format(arch),
            srcs = native.glob(["caffe2/perfkernels/*_{}.cc".format(arch)]),
            compatible_with = ["ovr_config//cpu:x86_64"],
            compiler_flags = select({
                "DEFAULT": [],
                "ovr_config//compiler:cl": [
                    "/arch:AVX2",
                    "/w",
                ],
                "ovr_config//compiler:clang": [
                    "-Wno-error",
                    "-mf16c",
                ] + (["-mf16c", "-mavx"] if arch == "avx" else ["-mfma", "-mavx2"] if arch == "avx2" else ["-mavx512f"]),
            }),
            raw_headers = native.glob([
                "caffe2/core/*.h",
                "caffe2/perfkernels/*.h",
                "caffe2/proto/*.h",
                "caffe2/utils/*.h",
            ], exclude = [
                "caffe2/core/macros.h",
            ]),
            reexport_all_header_dependencies = False,
            deps = [
                ":caffe2_proto_ovrsource",
                ":ovrsource_caffe2_macros.h",
                "@fbsource//xplat/caffe2/c10:c10_ovrsource",
            ],
        )
        for arch in ["avx", "avx2", "avx512"]
    ]

    oxx_static_library(
        name = "perfkernels_ovrsource",
        srcs = native.glob([
            "caffe2/perfkernels/*.cc",
        ], exclude = [
            "**/*_avx*",
        ]),
        compatible_with = cpu_supported_platforms,
        compiler_flags = select({
            "DEFAULT": [],
            "ovr_config//compiler:cl": [
                "/w",
            ],
            "ovr_config//compiler:clang": [
                "-Wno-macro-redefined",
                "-Wno-shadow",
                "-Wno-undef",
                "-Wno-unused-function",
                "-Wno-unused-local-typedef",
                "-Wno-unused-variable",
            ],
        }),
        public_include_directories = [],
        public_raw_headers = native.glob([
            "caffe2/perfkernels/*.h",
        ]),
        raw_headers = native.glob([
            "caffe2/core/*.h",
            "caffe2/proto/*.h",
            "caffe2/utils/*.h",
        ], exclude = [
            "caffe2/core/macros.h",
        ]),
        reexport_all_header_dependencies = False,
        deps = [
            ":caffe2_proto_ovrsource",
            ":ovrsource_caffe2_macros.h",
            "//third-party/cpuinfo:cpuinfo",
            "@fbsource//xplat/caffe2/c10:c10_ovrsource",
            "//third-party/protobuf:libprotobuf",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_64": [
                ":perfkernels_avx_ovrsource",
                ":perfkernels_avx2_ovrsource",
            ],
        }),
    )
