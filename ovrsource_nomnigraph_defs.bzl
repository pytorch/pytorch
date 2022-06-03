# @nolint
load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library", "oxx_test")
load("//arvr/tools/build_defs:oxx_python.bzl", "oxx_python_binary", "oxx_python_library")
load("//arvr/tools/build_defs:genrule_utils.bzl", "gen_cmake_header")
load("@bazel_skylib//lib:paths.bzl", "paths")

def define_nomnigraph():
    oxx_python_binary(
        name = "nomnigraph_gen_py_ovrsource",
        main_module = "caffe2.core.nomnigraph.op_gen",
        deps = [":nomnigraph_gen_py_main_ovrsource"],
    )

    oxx_python_library(
        name = "nomnigraph_gen_py_main_ovrsource",
        srcs = native.glob(["caffe2/core/nomnigraph/*.py"]),
        base_module = "",
    )

    nomnigraph_gen_py_cmd = " ".join([
        "--install_dir=$OUT",
        "--source_def=caffe2/core/nomnigraph/ops.def",
        # "--source_def=caffe2/core/nomnigraph/fb/ops.def",
    ])

    native.genrule(
        name = "nomnigraph_gen_ovrsource",
        srcs = [
            # "caffe2/core/nomnigraph/fb/ops.def",
            "caffe2/core/nomnigraph/op_gen.py",
            "caffe2/core/nomnigraph/ops.def",
        ],
        cmd_exe = "mkdir $OUT && $(exe :nomnigraph_gen_py_ovrsource) " + nomnigraph_gen_py_cmd,
        out = "gen",
    )

    TEST_SRCS = native.glob([
        "caffe2/core/nomnigraph/tests/*.cc",
    ], exclude = [
        "caffe2/core/nomnigraph/tests/GraphTest.cc",  # fails because debug iterator check
    ])

    oxx_static_library(
        name = "nomnigraph_ovrsource",
        srcs = [
            "caffe2/core/nomnigraph/Representations/NeuralNet.cc",
        ],
        compiler_flags = select({
            "ovr_config//compiler:clang": [
                "-Wno-undef",
                "-Wno-shadow",
                "-Wno-macro-redefined",
                "-Wno-unused-variable",
                "-Wno-unused-local-typedef",
                "-Wno-unused-function",
            ],
            "DEFAULT": [],
        }),
        public_include_directories = ["caffe2/core/nomnigraph/include"],
        public_raw_headers = native.glob([
            "caffe2/core/nomnigraph/include/**/*.h",
        ]),
        raw_headers = ["caffe2/core/common.h"],
        reexport_all_header_dependencies = False,
        tests = [
            ":" + paths.basename(filename)[:-len(".cc")] + "_ovrsource"
            for filename in TEST_SRCS
        ],
        deps = [
            ":ovrsource_caffe2_macros.h",
            "@fbsource//xplat/caffe2/c10:c10_ovrsource",
        ],
    )

    [
        oxx_test(
            name = paths.basename(filename)[:-len(".cc")] + "_ovrsource",
            srcs = [
                filename,
                "caffe2/core/nomnigraph/tests/test_util.cc",
            ],
            compiler_flags = select({
                "ovr_config//compiler:clang": [
                    "-Wno-macro-redefined",
                    "-Wno-shadow",
                    "-Wno-undef",
                    "-Wno-unused-variable",
                ],
                "DEFAULT": [],
            }),
            framework = "gtest",
            oncall = "frl_gemini",
            raw_headers = native.glob([
                "caffe2/core/nomnigraph/tests/*.h",
            ]),
            deps = [
                ":nomnigraph_ovrsource",
            ],
        )
        for filename in TEST_SRCS
    ]
