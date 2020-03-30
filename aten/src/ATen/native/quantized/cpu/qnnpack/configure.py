#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import confu
from confu import arm, x86


parser = confu.standard_parser()


def main(args):
    options = parser.parse_args(args)
    build = confu.Build.from_options(options)

    build.export_cpath("include", ["q8gemm.h"])

    with build.options(
        source_dir="src",
        deps=[
            build.deps.cpuinfo,
            build.deps.clog,
            build.deps.psimd,
            build.deps.fxdiv,
            build.deps.pthreadpool,
            build.deps.FP16,
        ],
        extra_include_dirs="src",
    ):

        requantization_objects = [
            build.cc("requantization/precise-scalar.c"),
            build.cc("requantization/fp32-scalar.c"),
            build.cc("requantization/q31-scalar.c"),
            build.cc("requantization/gemmlowp-scalar.c"),
        ]
        with build.options(isa=arm.neon if build.target.is_arm else None):
            requantization_objects += [
                build.cc("requantization/precise-psimd.c"),
                build.cc("requantization/fp32-psimd.c"),
            ]
        if build.target.is_x86 or build.target.is_x86_64:
            with build.options(isa=x86.sse2):
                requantization_objects += [
                    build.cc("requantization/precise-sse2.c"),
                    build.cc("requantization/fp32-sse2.c"),
                    build.cc("requantization/q31-sse2.c"),
                    build.cc("requantization/gemmlowp-sse2.c"),
                ]
            with build.options(isa=x86.ssse3):
                requantization_objects += [
                    build.cc("requantization/precise-ssse3.c"),
                    build.cc("requantization/q31-ssse3.c"),
                    build.cc("requantization/gemmlowp-ssse3.c"),
                ]
            with build.options(isa=x86.sse4_1):
                requantization_objects += [
                    build.cc("requantization/precise-sse4.c"),
                    build.cc("requantization/q31-sse4.c"),
                    build.cc("requantization/gemmlowp-sse4.c"),
                ]
        if build.target.is_arm or build.target.is_arm64:
            with build.options(isa=arm.neon if build.target.is_arm else None):
                requantization_objects += [
                    build.cc("requantization/precise-neon.c"),
                    build.cc("requantization/fp32-neon.c"),
                    build.cc("requantization/q31-neon.c"),
                    build.cc("requantization/gemmlowp-neon.c"),
                ]

        qnnpytorch_pack_objects = [
            # Common parts
            build.cc("init.c"),
            build.cc("operator-delete.c"),
            build.cc("operator-run.c"),
            # Operators
            build.cc("add.c"),
            build.cc("average-pooling.c"),
            build.cc("channel-shuffle.c"),
            build.cc("clamp.c"),
            build.cc("convolution.c"),
            build.cc("indirection.c"),
            build.cc("deconvolution.c"),
            build.cc("fully-connected.c"),
            build.cc("global-average-pooling.c"),
            build.cc("leaky-relu.c"),
            build.cc("max-pooling.c"),
            build.cc("sigmoid.c"),
            build.cc("softargmax.c"),
            build.cc("tanh.c"),
            # Scalar micro-kernels
            build.cc("u8lut32norm/scalar.c"),
            build.cc("x8lut/scalar.c"),
        ]

        with build.options(isa=arm.neon if build.target.is_arm else None):
            qnnpytorch_pack_objects += [
                build.cc("sconv/6x8-psimd.c"),
                build.cc("sdwconv/up4x9-psimd.c"),
                build.cc("sgemm/6x8-psimd.c"),
            ]

        with build.options(isa=arm.neon if build.target.is_arm else None):
            if build.target.is_arm or build.target.is_arm64:
                qnnpytorch_pack_objects += [
                    build.cc("q8avgpool/mp8x9p8q-neon.c"),
                    build.cc("q8avgpool/up8x9-neon.c"),
                    build.cc("q8avgpool/up8xm-neon.c"),
                    build.cc("q8conv/4x8-neon.c"),
                    build.cc("q8conv/8x8-neon.c"),
                    build.cc("q8dwconv/mp8x25-neon.c"),
                    build.cc("q8dwconv/up8x9-neon.c"),
                    build.cc("q8gavgpool/mp8x7p7q-neon.c"),
                    build.cc("q8gavgpool/up8x7-neon.c"),
                    build.cc("q8gavgpool/up8xm-neon.c"),
                    build.cc("q8gemm/4x-sumrows-neon.c"),
                    build.cc("q8gemm/4x8-neon.c"),
                    build.cc("q8gemm/4x8c2-xzp-neon.c"),
                    build.cc("q8gemm/6x4-neon.c"),
                    build.cc("q8gemm/8x8-neon.c"),
                    build.cc("q8vadd/neon.c"),
                    build.cc("sgemm/5x8-neon.c"),
                    build.cc("sgemm/6x8-neon.c"),
                    build.cc("u8clamp/neon.c"),
                    build.cc("u8maxpool/16x9p8q-neon.c"),
                    build.cc("u8maxpool/sub16-neon.c"),
                    build.cc("u8rmax/neon.c"),
                    build.cc("x8zip/x2-neon.c"),
                    build.cc("x8zip/x3-neon.c"),
                    build.cc("x8zip/x4-neon.c"),
                    build.cc("x8zip/xm-neon.c"),
                ]
            if build.target.is_arm:
                qnnpytorch_pack_objects += [
                    build.cc("hgemm/8x8-aarch32-neonfp16arith.S"),
                    build.cc("q8conv/4x8-aarch32-neon.S"),
                    build.cc("q8dwconv/up8x9-aarch32-neon.S"),
                    build.cc("q8gemm/4x8-aarch32-neon.S"),
                    build.cc("q8gemm/4x8c2-xzp-aarch32-neon.S"),
                ]
            if build.target.is_arm64:
                qnnpytorch_pack_objects += [
                    build.cc("q8gemm/8x8-aarch64-neon.S"),
                    build.cc("q8conv/8x8-aarch64-neon.S"),
                ]
            if build.target.is_x86 or build.target.is_x86_64:
                with build.options(isa=x86.sse2):
                    qnnpytorch_pack_objects += [
                        build.cc("q8avgpool/mp8x9p8q-sse2.c"),
                        build.cc("q8avgpool/up8x9-sse2.c"),
                        build.cc("q8avgpool/up8xm-sse2.c"),
                        build.cc("q8conv/4x4c2-sse2.c"),
                        build.cc("q8dwconv/mp8x25-sse2.c"),
                        build.cc("q8dwconv/up8x9-sse2.c"),
                        build.cc("q8gavgpool/mp8x7p7q-sse2.c"),
                        build.cc("q8gavgpool/up8x7-sse2.c"),
                        build.cc("q8gavgpool/up8xm-sse2.c"),
                        build.cc("q8gemm/2x4c8-sse2.c"),
                        build.cc("q8gemm/4x4c2-sse2.c"),
                        build.cc("q8vadd/sse2.c"),
                        build.cc("u8clamp/sse2.c"),
                        build.cc("u8maxpool/16x9p8q-sse2.c"),
                        build.cc("u8maxpool/sub16-sse2.c"),
                        build.cc("u8rmax/sse2.c"),
                        build.cc("x8zip/x2-sse2.c"),
                        build.cc("x8zip/x3-sse2.c"),
                        build.cc("x8zip/x4-sse2.c"),
                        build.cc("x8zip/xm-sse2.c"),
                    ]
            build.static_library("qnnpack", qnnpytorch_pack_objects)

    with build.options(
        source_dir="test",
        deps={
            (
                build,
                build.deps.cpuinfo,
                build.deps.clog,
                build.deps.pthreadpool,
                build.deps.FP16,
                build.deps.googletest,
            ): any,
            "log": build.target.is_android,
        },
        extra_include_dirs=["src", "test"],
    ):

        build.unittest("hgemm-test", build.cxx("hgemm.cc"))
        build.unittest("q8avgpool-test", build.cxx("q8avgpool.cc"))
        build.unittest("q8conv-test", build.cxx("q8conv.cc"))
        build.unittest("q8dwconv-test", build.cxx("q8dwconv.cc"))
        build.unittest("q8gavgpool-test", build.cxx("q8gavgpool.cc"))
        build.unittest("q8gemm-test", build.cxx("q8gemm.cc"))
        build.unittest("q8vadd-test", build.cxx("q8vadd.cc"))
        build.unittest("sconv-test", build.cxx("sconv.cc"))
        build.unittest("sgemm-test", build.cxx("sgemm.cc"))
        build.unittest("u8clamp-test", build.cxx("u8clamp.cc"))
        build.unittest("u8lut32norm-test", build.cxx("u8lut32norm.cc"))
        build.unittest("u8maxpool-test", build.cxx("u8maxpool.cc"))
        build.unittest("u8rmax-test", build.cxx("u8rmax.cc"))
        build.unittest("x8lut-test", build.cxx("x8lut.cc"))
        build.unittest("x8zip-test", build.cxx("x8zip.cc"))

        build.unittest("add-test", build.cxx("add.cc"))
        build.unittest("average-pooling-test", build.cxx("average-pooling.cc"))
        build.unittest("channel-shuffle-test", build.cxx("channel-shuffle.cc"))
        build.unittest("clamp-test", build.cxx("clamp.cc"))
        build.unittest("convolution-test", build.cxx("convolution.cc"))
        build.unittest("deconvolution-test", build.cxx("deconvolution.cc"))
        build.unittest("fully-connected-test", build.cxx("fully-connected.cc"))
        build.unittest(
            "global-average-pooling-test", build.cxx("global-average-pooling.cc")
        )
        build.unittest("leaky-relu-test", build.cxx("leaky-relu.cc"))
        build.unittest("max-pooling-test", build.cxx("max-pooling.cc"))
        build.unittest("sigmoid-test", build.cxx("sigmoid.cc"))
        build.unittest("softargmax-test", build.cxx("softargmax.cc"))
        build.unittest("tanh-test", build.cxx("tanh.cc"))
        build.unittest(
            "requantization-test",
            [build.cxx("requantization.cc")] + requantization_objects,
        )

    benchmark_isa = None
    if build.target.is_arm:
        benchmark_isa = arm.neon
    elif build.target.is_x86:
        benchmark_isa = x86.sse4_1
    with build.options(
        source_dir="bench",
        deps={
            (
                build,
                build.deps.cpuinfo,
                build.deps.clog,
                build.deps.pthreadpool,
                build.deps.FP16,
                build.deps.googlebenchmark,
            ): any,
            "log": build.target.is_android,
        },
        isa=benchmark_isa,
        extra_include_dirs="src",
    ):

        build.benchmark("add-bench", build.cxx("add.cc"))
        build.benchmark("average-pooling-bench", build.cxx("average-pooling.cc"))
        build.benchmark("channel-shuffle-bench", build.cxx("channel-shuffle.cc"))
        build.benchmark("convolution-bench", build.cxx("convolution.cc"))
        build.benchmark(
            "global-average-pooling-bench", build.cxx("global-average-pooling.cc")
        )
        build.benchmark("max-pooling-bench", build.cxx("max-pooling.cc"))
        build.benchmark("sigmoid-bench", build.cxx("sigmoid.cc"))
        build.benchmark("softargmax-bench", build.cxx("softargmax.cc"))
        build.benchmark("tanh-bench", build.cxx("tanh.cc"))

        build.benchmark("q8gemm-bench", build.cxx("q8gemm.cc"))
        build.benchmark("hgemm-bench", build.cxx("hgemm.cc"))
        build.benchmark("sgemm-bench", build.cxx("sgemm.cc"))
        build.benchmark(
            "requantization-bench",
            [build.cxx("requantization.cc")] + requantization_objects,
        )

    return build


if __name__ == "__main__":
    import sys

    main(sys.argv[1:]).generate()
