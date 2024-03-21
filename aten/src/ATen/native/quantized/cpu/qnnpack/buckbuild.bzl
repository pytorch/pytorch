load("//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("//tools/build_defs:fb_xplat_cxx_test.bzl", "fb_xplat_cxx_test")
load("//tools/build_defs:glob_defs.bzl", "subdir_glob")
load("//tools/build_defs:platform_defs.bzl", "ANDROID", "APPLE", "APPLETVOS", "CXX", "IOS", "MACOSX")

# Shared by internal and OSS BUCK
def define_qnnpack(third_party, labels = []):
    fb_xplat_cxx_library(
        # @autodeps-skip
        name = "ukernels_scalar",
        srcs = [
            "src/requantization/fp32-scalar.c",
            "src/requantization/gemmlowp-scalar.c",
            "src/requantization/precise-scalar.c",
            "src/requantization/q31-scalar.c",
            "src/u8lut32norm/scalar.c",
            "src/x8lut/scalar.c",
        ],
        headers = subdir_glob([
            ("src", "qnnpack/*.h"),
            ("src", "requantization/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
        ],
        fbobjc_preprocessor_flags = [
            "-DQNNP_PRIVATE=",
            "-DQNNP_INTERNAL=",
        ],
        force_static = True,
        labels = labels,
        visibility = ["PUBLIC"],
        deps = [
            ":qnnp_interface",
            third_party("cpuinfo"),
            third_party("FP16"),
            third_party("FXdiv"),
        ],
    )

    fb_xplat_cxx_library(
        # @autodeps-skip
        name = "ukernels_sse2",
        srcs = [
            "wrappers/q8avgpool/mp8x9p8q-sse2.c",
            "wrappers/q8avgpool/up8x9-sse2.c",
            "wrappers/q8avgpool/up8xm-sse2.c",
            "wrappers/q8conv/4x4c2-sse2.c",
            "wrappers/q8dwconv/mp8x25-sse2.c",
            "wrappers/q8dwconv/mp8x25-sse2-per-channel.c",
            "wrappers/q8dwconv/mp8x27-sse2.c",
            "wrappers/q8dwconv/up8x9-sse2.c",
            "wrappers/q8dwconv/up8x9-sse2-per-channel.c",
            "wrappers/q8gavgpool/mp8x7p7q-sse2.c",
            "wrappers/q8gavgpool/up8x7-sse2.c",
            "wrappers/q8gavgpool/up8xm-sse2.c",
            "wrappers/q8gemm/2x4c8-sse2.c",
            "wrappers/q8gemm/4x4c2-dq-sse2.c",
            "wrappers/q8gemm/4x4c2-sse2.c",
            "wrappers/q8gemm_sparse/8x4c1x4-packed-sse2.c",
            "wrappers/q8vadd/sse2.c",
            "wrappers/requantization/fp32-sse2.c",
            "wrappers/requantization/gemmlowp-sse2.c",
            "wrappers/requantization/precise-sse2.c",
            "wrappers/requantization/q31-sse2.c",
            "wrappers/u8clamp/sse2.c",
            "wrappers/u8maxpool/16x9p8q-sse2.c",
            "wrappers/u8maxpool/sub16-sse2.c",
            "wrappers/u8rmax/sse2.c",
            "wrappers/x8zip/x2-sse2.c",
            "wrappers/x8zip/x3-sse2.c",
            "wrappers/x8zip/x4-sse2.c",
            "wrappers/x8zip/xm-sse2.c",
        ],
        headers = subdir_glob([
            ("src", "**/*.c"),
            ("src", "q8gemm_sparse/*.h"),
            ("src", "qnnpack/*.h"),
            ("src", "requantization/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O3",
            "-ffast-math",
            "-Wno-error=unused-variable",
            "-Wno-shadow",
            "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
            "-Wno-empty-translation-unit",
        ],
        fbobjc_preprocessor_flags = [
            "-DQNNP_PRIVATE=",
            "-DQNNP_INTERNAL=",
        ],
        force_static = True,
        labels = labels,
        platform_compiler_flags = [
            (
                "86",
                [
                    "-msse2",
                    "-mno-sse3",
                ],
            ),
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":qnnp_interface",
            third_party("cpuinfo"),
            third_party("FP16"),
            third_party("FXdiv"),
        ],
    )

    fb_xplat_cxx_library(
        # @autodeps-skip
        name = "ukernels_ssse3",
        srcs = [
            "wrappers/requantization/gemmlowp-ssse3.c",
            "wrappers/requantization/precise-ssse3.c",
            "wrappers/requantization/q31-ssse3.c",
        ],
        headers = subdir_glob([
            ("src", "**/*.c"),
            ("src", "qnnpack/*.h"),
            ("src", "requantization/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O3",
            "-ffast-math",
            "-Wno-error=unused-variable",
            "-Wno-shadow",
            "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
            "-Wno-empty-translation-unit",
        ],
        fbobjc_preprocessor_flags = [
            "-DQNNP_PRIVATE=",
            "-DQNNP_INTERNAL=",
        ],
        force_static = True,
        labels = labels,
        platform_compiler_flags = [
            (
                "86",
                [
                    "-mssse3",
                    "-mno-sse4",
                ],
            ),
            (
                # By default, osmeta compiler silently ignores -msseXX flags.
                # This flag disables this behavior.
                "osmeta",
                [
                    "-mosmeta-no-restrict-sse",
                ],
            ),
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":qnnp_interface",
            third_party("cpuinfo"),
            third_party("FP16"),
            third_party("FXdiv"),
        ],
    )

    fb_xplat_cxx_library(
        # @autodeps-skip
        name = "ukernels_sse41",
        srcs = [
            "wrappers/requantization/gemmlowp-sse4.c",
            "wrappers/requantization/precise-sse4.c",
            "wrappers/requantization/q31-sse4.c",
        ],
        headers = subdir_glob([
            ("src", "**/*.c"),
            ("src", "qnnpack/*.h"),
            ("src", "requantization/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O3",
            "-ffast-math",
            "-Wno-error=unused-variable",
            "-Wno-shadow",
            "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
            "-Wno-empty-translation-unit",
        ],
        fbobjc_preprocessor_flags = [
            "-DQNNP_PRIVATE=",
            "-DQNNP_INTERNAL=",
        ],
        force_static = True,
        labels = labels,
        platform_compiler_flags = [
            (
                "86",
                [
                    "-msse4.1",
                    "-mno-sse4.2",
                ],
            ),
            (
                # By default, osmeta compiler silently ignores -msseXX flags.
                # This flag disables this behavior.
                "osmeta",
                [
                    "-mosmeta-no-restrict-sse",
                ],
            ),
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":qnnp_interface",
            third_party("cpuinfo"),
            third_party("FP16"),
            third_party("FXdiv"),
        ],
    )

    fb_xplat_cxx_library(
        # @autodeps-skip
        name = "qnnp_interface",
        headers = subdir_glob(
            [
                ("include", "*.h"),
                ("src", "qnnpack/*.h"),
                ("src", "requantization/*.h"),
            ],
        ),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
        ],
        force_static = True,
        labels = labels,
        visibility = ["PUBLIC"],
        deps = [
            third_party("pthreadpool_header"),
        ],
    )

    fb_xplat_cxx_library(
        # @autodeps-skip
        name = "pytorch_qnnpack",
        srcs = [
            "src/add.c",
            "src/average-pooling.c",
            "src/channel-shuffle.c",
            "src/clamp.c",
            "src/conv-prepack.cc",
            "src/conv-run.cc",
            "src/convolution.c",
            "src/deconv-run.cc",
            "src/deconvolution.c",
            "src/fc-dynamic-run.cc",
            "src/fc-prepack.cc",
            "src/fc-run.cc",
            "src/fc-unpack.cc",
            "src/fully-connected.c",
            "src/fully-connected-sparse.c",
            "src/global-average-pooling.c",
            "src/hardsigmoid.c",
            "src/hardswish.c",
            "src/indirection.c",
            "src/init.c",
            "src/leaky-relu.c",
            "src/max-pooling.c",
            "src/operator-delete.c",
            "src/operator-run.c",
            "src/sigmoid.c",
            "src/softargmax.c",
            "src/tanh.c",
        ],
        headers = subdir_glob([
            ("src", "**/*.c"),
            ("src", "**/*.h"),
            ("src", "qnnpack/*.h"),
            ("include", "**/*.h"),
        ]),
        header_namespace = "",
        exported_headers = subdir_glob([
            ("src", "qnnpack/*.h"),
            ("include", "*.h"),
        ]),
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
        ],
        fbobjc_preprocessor_flags = [
            "-DQNNP_PRIVATE=",
            "-DQNNP_INTERNAL=",
        ],
        labels = [
            "supermodule:android/default/pytorch",
            "supermodule:ios/default/public.pytorch",
        ],
        platform_compiler_flags = [
            (
                "armv7",
                [
                    "-mfpu=neon",
                ],
            ),
            (
                "^android-armv7$",
                [
                    "-marm",
                    "-mfloat-abi=softfp",
                ],
            ),
        ],
        # FIXME(T172572183): This should be removed when fbcode no longer uses
        # produce_interface_from_stub_shared_library; it's needed to work around a bug
        # in that mode.
        supports_shlib_interfaces = select({
            "ovr_config//os:linux": False,
            "DEFAULT": True,
        }),
        visibility = ["PUBLIC"],
        deps = [
            ":qnnp_interface",
            ":ukernels_asm",
            ":ukernels_neon",
            ":ukernels_psimd",
            ":ukernels_scalar",
            ":ukernels_sse2",
            ":ukernels_sse41",
            ":ukernels_ssse3",
            third_party("clog"),
            third_party("cpuinfo"),
            third_party("FP16"),
            third_party("FXdiv"),
            third_party("pthreadpool"),
        ],
        exported_deps = [
            third_party("cpuinfo"),
        ],
    )

    # Only ukernels implemented in C with ARM NEON intrinsics
    fb_xplat_cxx_library(
        # @autodeps-skip
        name = "ukernels_neon",
        srcs = [
            "wrappers/q8avgpool/mp8x9p8q-neon.c",
            "wrappers/q8avgpool/up8x9-neon.c",
            "wrappers/q8avgpool/up8xm-neon.c",
            "wrappers/q8conv/4x8-neon.c",
            "wrappers/q8conv/8x8-neon.c",
            "wrappers/q8dwconv/mp8x25-neon.c",
            "wrappers/q8dwconv/mp8x25-neon-per-channel.c",
            "wrappers/q8dwconv/mp8x27-neon.c",
            "wrappers/q8dwconv/up8x9-neon.c",
            "wrappers/q8dwconv/up8x9-neon-per-channel.c",
            "wrappers/q8gavgpool/mp8x7p7q-neon.c",
            "wrappers/q8gavgpool/up8x7-neon.c",
            "wrappers/q8gavgpool/up8xm-neon.c",
            "wrappers/q8gemm/4x-sumrows-neon.c",
            "wrappers/q8gemm/4x8-dq-neon.c",
            "wrappers/q8gemm/4x8-neon.c",
            "wrappers/q8gemm/4x8c2-xzp-neon.c",
            "wrappers/q8gemm/6x4-neon.c",
            "wrappers/q8gemm/8x8-neon.c",
            "wrappers/q8vadd/neon.c",
            "wrappers/requantization/fp32-neon.c",
            "wrappers/requantization/gemmlowp-neon.c",
            "wrappers/requantization/precise-neon.c",
            "wrappers/requantization/q31-neon.c",
            "wrappers/sgemm/5x8-neon.c",
            "wrappers/sgemm/6x8-neon.c",
            "wrappers/u8clamp/neon.c",
            "wrappers/u8maxpool/16x9p8q-neon.c",
            "wrappers/u8maxpool/sub16-neon.c",
            "wrappers/u8rmax/neon.c",
            "wrappers/x8zip/x2-neon.c",
            "wrappers/x8zip/x3-neon.c",
            "wrappers/x8zip/x4-neon.c",
            "wrappers/x8zip/xm-neon.c",
        ],
        headers = subdir_glob([
            ("src", "**/*.c"),
            ("src", "qnnpack/*.h"),
            ("src", "requantization/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O3",
            "-ffast-math",
            "-Wno-error=unused-variable",
            "-Wno-shadow",
            "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
        ],
        fbobjc_preprocessor_flags = [
            "-DQNNP_PRIVATE=",
            "-DQNNP_INTERNAL=",
        ],
        force_static = True,
        labels = labels,
        platform_compiler_flags = [
            (
                "armv7",
                [
                    "-mfpu=neon",
                ],
            ),
            (
                "^android-armv7$",
                [
                    "-marm",
                    "-mfloat-abi=softfp",
                ],
            ),
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":qnnp_interface",
            third_party("cpuinfo"),
            third_party("FP16"),
            third_party("FXdiv"),
        ],
    )

    fb_xplat_cxx_library(
        # @autodeps-skip
        name = "ukernels_asm",
        srcs = [
            # Dummy empty source file to work around link error on x86-64 Android
            # when static library contains no symbols.
            "wrappers/dummy.c",
            # AArch32 ukernels
            "wrappers/hgemm/8x8-aarch32-neonfp16arith.S",
            "wrappers/q8conv/4x8-aarch32-neon.S",
            "wrappers/q8dwconv/up8x9-aarch32-neon.S",
            "wrappers/q8dwconv/up8x9-aarch32-neon-per-channel.S",
            "wrappers/q8gemm/4x8-aarch32-neon.S",
            "wrappers/q8gemm/4x8-dq-aarch32-neon.S",
            "wrappers/q8gemm/4x8c2-xzp-aarch32-neon.S",
            "wrappers/q8gemm_sparse/4x4-packA-aarch32-neon.S",
            "wrappers/q8gemm_sparse/4x8c1x4-dq-packedA-aarch32-neon.S",
            "wrappers/q8gemm_sparse/4x8c8x1-dq-packedA-aarch32-neon.S",
            "wrappers/q8gemm_sparse/8x4-packA-aarch64-neon.S",
            "wrappers/q8gemm_sparse/8x8c1x4-dq-packedA-aarch64-neon.S",
            "wrappers/q8gemm_sparse/8x8c8x1-dq-packedA-aarch64-neon.S",
            # AArch64 ukernels
            "wrappers/q8conv/8x8-aarch64-neon.S",
            "wrappers/q8gemm/8x8-aarch64-neon.S",
            "wrappers/q8gemm/8x8-dq-aarch64-neon.S",
        ],
        headers = subdir_glob([
            ("src", "qnnpack/assembly.h"),
            ("src", "**/*.S"),
            ("src", "requantization/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
        ],
        fbobjc_preprocessor_flags = [
            "-DQNNP_PRIVATE=",
            "-DQNNP_INTERNAL=",
        ],
        force_static = True,
        labels = labels,
        platform_compiler_flags = [
            (
                # iOS assembler doesn't let us specify ISA in the assembly file,
                # so this must be set to the highest version of ISA of any of the
                # assembly functions
                "^iphoneos-armv7$",
                [
                    "-mfpu=neon-vfpv4",
                ],
            ),
            (
                "osmeta",
                [
                    "-mfpu=neon-vfpv4",
                ],
            ),
        ],
        platform_preprocessor_flags = [
            (
                "android",
                [
                    # Workaround for osmeta-android, which builds for ELF, but hides it
                    "-D__ELF__=1",
                ],
            ),
            (
                "tizen",
                [
                    # Workaround for osmeta-tizen, which builds for ELF, but hides it
                    "-D__ELF__=1",
                ],
            ),
        ],
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        # @autodeps-skip
        name = "ukernels_psimd",
        srcs = [
            "src/requantization/fp32-psimd.c",
            "src/requantization/precise-psimd.c",
            "src/sgemm/6x8-psimd.c",
        ],
        headers = subdir_glob([
            ("src", "**/*.c"),
            ("src", "qnnpack/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O3",
            "-ffast-math",
            "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
        ],
        fbobjc_preprocessor_flags = [
            "-DQNNP_PRIVATE=",
            "-DQNNP_INTERNAL=",
        ],
        force_static = True,
        labels = labels,
        platform_compiler_flags = [
            (
                "armv7",
                [
                    "-mfpu=neon",
                ],
            ),
            (
                "^android-armv7$",
                [
                    "-marm",
                    "-mfloat-abi=softfp",
                ],
            ),
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":qnnp_interface",
            third_party("cpuinfo"),
            third_party("FP16"),
            third_party("FXdiv"),
            third_party("psimd"),
        ],
    )

    fb_xplat_cxx_test(
        # @autodeps-skip
        fbandroid_use_instrumentation_test = True,
        contacts = ["oncall+ai_infra_mobile_platform@xmail.facebook.com"],
        platforms = (CXX, APPLE, ANDROID),
        apple_sdks = (IOS, MACOSX),
        name = "pytorch_qnnpack_test",
        srcs = [
            "test/add.cc",
            "test/average-pooling.cc",
            "test/channel-shuffle.cc",
            "test/clamp.cc",
            "test/convolution.cc",
            "test/deconvolution.cc",
            "test/fully-connected.cc",
            "test/fully-connected-sparse.cc",
            "test/global-average-pooling.cc",
            "test/hardsigmoid.cc",
            "test/hardswish.cc",
            "test/leaky-relu.cc",
            "test/max-pooling.cc",
            "test/q8avgpool.cc",
            "test/q8conv.cc",
            "test/q8dwconv.cc",
            "test/q8gavgpool.cc",
            "test/q8gemm_sparse.cc",
            "test/q8vadd.cc",
            "test/requantization.cc",
            "test/sgemm.cc",
            "test/sigmoid.cc",
            "test/softargmax.cc",
            "test/tanh.cc",
            "test/u8clamp.cc",
            "test/u8lut32norm.cc",
            "test/u8maxpool.cc",
            "test/u8rmax.cc",
            "test/x8lut.cc",
            "test/x8zip.cc",
        ],
        headers = {
            "add-operator-tester.h": "test/add-operator-tester.h",
            "average-pooling-operator-tester.h": "test/average-pooling-operator-tester.h",
            "avgpool-microkernel-tester.h": "test/avgpool-microkernel-tester.h",
            "channel-shuffle-operator-tester.h": "test/channel-shuffle-operator-tester.h",
            "clamp-microkernel-tester.h": "test/clamp-microkernel-tester.h",
            "clamp-operator-tester.h": "test/clamp-operator-tester.h",
            "convolution-operator-tester.h": "test/convolution-operator-tester.h",
            "deconvolution-operator-tester.h": "test/deconvolution-operator-tester.h",
            "dwconv-microkernel-tester.h": "test/dwconv-microkernel-tester.h",
            "fully-connected-operator-tester.h": "test/fully-connected-operator-tester.h",
            "fully-connected-sparse-operator-tester.h": "test/fully-connected-sparse-operator-tester.h",
            "gavgpool-microkernel-tester.h": "test/gavgpool-microkernel-tester.h",
            "gemm-block-sparse-microkernel-tester.h": "test/gemm-block-sparse-microkernel-tester.h",
            "gemm-microkernel-tester.h": "test/gemm-microkernel-tester.h",
            "global-average-pooling-operator-tester.h": "test/global-average-pooling-operator-tester.h",
            "hardsigmoid-operator-tester.h": "test/hardsigmoid-operator-tester.h",
            "hardswish-operator-tester.h": "test/hardswish-operator-tester.h",
            "leaky-relu-operator-tester.h": "test/leaky-relu-operator-tester.h",
            "lut-microkernel-tester.h": "test/lut-microkernel-tester.h",
            "lut-norm-microkernel-tester.h": "test/lut-norm-microkernel-tester.h",
            "max-pooling-operator-tester.h": "test/max-pooling-operator-tester.h",
            "maxpool-microkernel-tester.h": "test/maxpool-microkernel-tester.h",
            "requantization-tester.h": "test/requantization-tester.h",
            "rmax-microkernel-tester.h": "test/rmax-microkernel-tester.h",
            "sigmoid-operator-tester.h": "test/sigmoid-operator-tester.h",
            "softargmax-operator-tester.h": "test/softargmax-operator-tester.h",
            "tanh-operator-tester.h": "test/tanh-operator-tester.h",
            "test_utils.h": "test/test_utils.h",
            "vadd-microkernel-tester.h": "test/vadd-microkernel-tester.h",
            "zip-microkernel-tester.h": "test/zip-microkernel-tester.h",
        },
        header_namespace = "",
        compiler_flags = [
            "-fexceptions",
            "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
        ],
        platform_linker_flags = [
            (
                "^linux.*$",
                [
                    "-Wl,--no-as-needed",
                    "-ldl",
                    "-pthread",
                ],
            ),
        ],
        env = {
            # These tests fail in sandcastle since they leak memory. Disable LeakSanitizer.
            "ASAN_OPTIONS": "detect_leaks=0",
        },
        deps = [
            ":pytorch_qnnpack",
            ":ukernels_asm",
            ":ukernels_neon",
            ":ukernels_psimd",
            ":ukernels_scalar",
            ":ukernels_sse2",
            ":ukernels_sse41",
            ":ukernels_ssse3",
            third_party("cpuinfo"),
            third_party("FP16"),
            third_party("pthreadpool"),
        ],
    )
