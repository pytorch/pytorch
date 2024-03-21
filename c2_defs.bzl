load("@bazel_skylib//lib:collections.bzl", "collections")
load("@bazel_skylib//lib:paths.bzl", "paths")
load("@fbcode_macros//build_defs:native_rules.bzl", "buck_genrule")
load("@fbsource//tools/build_defs:default_platform_defs.bzl", "compose_platform_setting_list")
load("@fbsource//tools/build_defs:dict_defs.bzl", "dict_defs")
load("@fbsource//tools/build_defs:expect.bzl", "expect")
load("@fbsource//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_arvr_mode", "is_fbcode_mode_mac")
load("@fbsource//tools/build_defs:platform_defs.bzl", "ANDROID", "APPLE", "CXX", "IOS", "MACOSX", "WINDOWS")
load("@fbsource//tools/build_defs/apple:build_mode_defs.bzl", "is_production_build")
load("@fbsource//tools/build_defs/apple:config_utils_defs.bzl", "STATIC_LIBRARY_IOS_CONFIG", "STATIC_LIBRARY_MAC_CONFIG", "fbobjc_configs")
load("@fbsource//xplat/caffe2:buckbuild.bzl", "read_bool")
load("@fbsource//xplat/pfh/Msgr/Mobile/ProductInfra:DEFS.bzl", "Msgr_Mobile_ProductInfra")

def get_c2_expose_op_to_c10():
    c2_op_to_c10 = native.read_config("caffe2", "expose_op_to_c10", "0")

    expect(
        c2_op_to_c10 in ("0", "1"),
        c2_op_to_c10,
    )

    return bool(int(c2_op_to_c10))

def get_c2_mpscnn():
    c2_mpscnn = native.read_config("caffe2", "enable_mpscnn", "1")

    expect(
        c2_mpscnn in ("0", "1"),
        c2_mpscnn,
    )

    return bool(int(c2_mpscnn))

def get_c2_mpscnn_test():
    c2_mpscnn_test = native.read_config("caffe2", "enable_mpscnn_test", "0")

    expect(
        c2_mpscnn_test in ("0", "1"),
        c2_mpscnn_test,
    )

    return bool(int(c2_mpscnn_test))

def get_c2_nomnigraph():
    c2_nomnigraph = native.read_config("caffe2", "enable_nomnigraph", "1")

    expect(
        c2_nomnigraph in ("0", "1"),
        c2_nomnigraph,
    )

    return bool(int(c2_nomnigraph))

def get_c2_qpl():
    c2_qpl = native.read_config("caffe2", "enable_qpl", "1")

    expect(
        c2_qpl in ("0", "1"),
        c2_qpl,
    )

    return bool(int(c2_qpl))

def get_c2_strip_debug_info():
    c2_strip_debug_info = native.read_config("caffe2", "strip_debug_info", "0")

    expect(
        c2_strip_debug_info in ("0", "1"),
        c2_strip_debug_info,
    )

    return bool(int(c2_strip_debug_info))

def get_c2_strip_glog():
    c2_strip_glog = native.read_config("caffe2", "strip_glog", "1")

    expect(
        c2_strip_glog in ("0", "1"),
        c2_strip_glog,
    )

    return bool(int(c2_strip_glog))

def get_c2_tvm():
    c2_tvm = native.read_config("caffe2", "enable_tvm", "1")

    expect(
        c2_tvm in ("0", "1"),
        c2_tvm,
    )

    return bool(int(c2_tvm))

_C2_XPLAT_NO_HPTT_PREPROCESSOR_FLAGS = [
    "-Icaffe2",
    "-Imodules",
    "-DEIGEN_NO_DEBUG",
    "-DCAFFE2_USE_LITE_PROTO",
    "-DCAFFE2_USE_GOOGLE_GLOG",
    "-DCAFFE2_RNN_NO_TEXT_FORMAT",
    "-DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK=1",
    "-DCAFFE2_IS_XPLAT_BUILD",
    "-DSTRIP_ERROR_MESSAGES",
    "-DUSE_INTERNAL_PTHREADPOOL_IMPL",
]

def get_c2_xplat_no_hptt_preprocessor_flags():
    flags = []
    flags += _C2_XPLAT_NO_HPTT_PREPROCESSOR_FLAGS
    if is_arvr_mode() and get_c2_strip_glog():
        flags += ["-UGOOGLE_STRIP_LOG", "-DGOOGLE_STRIP_LOG=1"]
    if get_c2_expose_op_to_c10():
        flags += ["-DEXPOSE_C2_OPS", "-frtti"]
    return flags

C2_XPLAT_SERVER_PREPROCESSOR_FLAGS = [
    "-DCAFFE2_USE_EIGEN_FOR_BLAS",
    "-DC10_DISABLE_SIGNAL_HANDLERS",
    "-DCAFFE2_DISABLE_NUMA",
]

C2_XPLAT_HPTT_PREPROCESSOR_FLAGS = [
    "-DCAFFE2_USE_HPTT",
]

def get_c2_xplat_preprocessor_flags():
    flags = get_c2_xplat_no_hptt_preprocessor_flags() + C2_XPLAT_HPTT_PREPROCESSOR_FLAGS
    if get_c2_nomnigraph():
        flags.append("-DCAFFE2_OPTIMIZER")
    return flags

def get_c2_xplat_no_hptt_compiler_flags():
    return [
        "-Os",
        "-fexceptions",
        "-frtti",
        "-Wno-shadow",
        "-Wno-unknown-pragmas",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
    ]

def get_c2_xplat_compiler_flags():
    return get_c2_xplat_no_hptt_compiler_flags() + C2_XPLAT_HPTT_PREPROCESSOR_FLAGS

def get_c2_fbobjc_xplat_compiler_flags():
    flags = []

    if is_production_build():
        flags.append("-DCAFFE2_NO_OPERATOR_SCHEMA")

    flags.append("-DCAFFE2_NO_GRADIENT_OPS")

    # For iOS production builds (and all Android builds), strip GLOG logging to
    # save size. We can disable by setting caffe2.strip_glog=0 in .buckconfig.local.
    if is_production_build() or get_c2_strip_glog():
        flags += ["-UGOOGLE_STRIP_LOG", "-DGOOGLE_STRIP_LOG=3"]
    else:
        flags.append("-UGOOGLE_STRIP_LOG")

    return flags

def get_c2_fbandroid_xplat_compiler_flags():
    flags = [
        "-Wno-unused-but-set-variable",
        "-DHAVE_MMAP",
    ]

    if get_c2_strip_glog():
        flags += ["-UGOOGLE_STRIP_LOG", "-DGOOGLE_STRIP_LOG=1"]

    if get_c2_strip_debug_info():
        flags.append("-g0")

    return flags

_C2_FBOBJC_COMPILER_FLAGS = [
    "-Wno-missing-prototypes",
    "-Wno-global-constructors",
    "-Wno-unknown-pragmas",
    "-Wno-invalid-partial-specialization",
    "-Wno-missing-braces",
    "-Wno-range-loop-analysis",
]

def get_c2_fbobjc_compiler_flags():
    flags = list(_C2_FBOBJC_COMPILER_FLAGS)

    # Avoid linking Accelerate on MacOS because we have
    # inconsistent LAPACK headers (see problems in D19257077).
    flags.append("-DCAFFE2_USE_ACCELERATE" if not is_arvr_mode() else "-DCAFFE2_USE_EIGEN_FOR_BLAS")
    if get_c2_mpscnn():
        flags.append(
            # TODO(t19120552) - fix this. MPSCNNConvolutionDescriptor.strideInPixelsX
            # is marked as iOS 11+, but it's been available since iOS 10.
            "-Wno-unguarded-availability",
        )
    return flags

C2_FBOBJC_MACOSX_COMPILER_FLAGS = [
    "-msse4.2",
]

C2_FBOBJC_IPHONE_COMPILER_FLAGS = [
    "-mfpu=neon-fp16",
]

def get_c2_fbobjc_frameworks():
    frameworks = []
    if not is_arvr_mode():
        frameworks.append(
            # On iOS, presumably Accelerate is a faster BLAS
            "$SDKROOT/System/Library/Frameworks/Accelerate.framework",
        )
    return frameworks

def get_c2_fbobjc_ios_frameworks():
    frameworks = []

    if get_c2_mpscnn():
        frameworks.extend([
            "$SDKROOT/System/Library/Frameworks/Metal.framework",
            "$SDKROOT/System/Library/Frameworks/MetalPerformanceShaders.framework",
        ])

    return frameworks

def get_c2_fbobjc_exported_preprocessor_flags():
    flags = []

    if get_c2_mpscnn():
        flags.append("-DCAFFE2_USE_MPSCNN")

        if get_c2_mpscnn_test():
            flags.append("-DCAFFE2_USE_MPSCNN_TEST")

    return flags

def get_c2_fbandroid_exported_preprocessor_flags():
    flags = []

    BUILD_MODE_DO_NOT_USE_WITHOUT_ASKING_SERIOUSLY = native.read_config(
        "fbandroid",
        "build_mode",
        "dev",
    )
    if BUILD_MODE_DO_NOT_USE_WITHOUT_ASKING_SERIOUSLY == "opt":
        flags.append("-DCAFFE2_NO_OPERATOR_SCHEMA")

    flags.append("-DCAFFE2_NO_GRADIENT_OPS")

    return flags

C2_FBANDROID_COMPILER_FLAGS = [
    "-DCAFFE2_USE_EIGEN_FOR_BLAS",
    "-Wno-unknown-pragmas",
    "-Wno-deprecated-declarations",
    "-Wno-invalid-partial-specialization",
    "-Wno-missing-braces",
]

C2_FBANDROID_ARMV7_COMPILER_FLAGS = [
    "-mfpu=neon-fp16",
]

C2_FBANDROID_X86_COMPILER_FLAGS = [
    "-mssse3",
]

C2_FBANDROID_LINKER_FLAGS = []

C2_FBOBJC_EXTRA_TARGET_CONFIG = {
    "MTL_LANGUAGE_REVISION": "Metal12",
}

def get_c2_default_cxx_args():
    return dict(
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = get_c2_xplat_compiler_flags(),
        fbandroid_compiler_flags = C2_FBANDROID_COMPILER_FLAGS + get_c2_fbandroid_xplat_compiler_flags(),
        fbandroid_exported_platform_preprocessor_flags = [
            (
                "android-armv7",
                get_c2_fbandroid_exported_preprocessor_flags(),
            ),
        ],
        fbandroid_linker_flags = C2_FBANDROID_LINKER_FLAGS,
        fbandroid_platform_compiler_flags = [
            ("android-armv7", C2_FBANDROID_ARMV7_COMPILER_FLAGS),
            (".*x86.*", C2_FBANDROID_X86_COMPILER_FLAGS),
        ],
        fbobjc_compiler_flags = get_c2_fbobjc_compiler_flags() + get_c2_fbobjc_xplat_compiler_flags(),
        fbobjc_configs = fbobjc_configs(
            STATIC_LIBRARY_IOS_CONFIG,
            extra_target_config = C2_FBOBJC_EXTRA_TARGET_CONFIG,
        ),
        fbobjc_exported_platform_preprocessor_flags = [
            (
                "iphoneos",
                get_c2_fbobjc_exported_preprocessor_flags(),
            ),
        ],
        fbobjc_frameworks = get_c2_fbobjc_frameworks() + get_c2_fbobjc_ios_frameworks(),
        fbobjc_platform_compiler_flags = [
            ("iphoneos", C2_FBOBJC_IPHONE_COMPILER_FLAGS),
        ],
        macosx_compiler_flags = C2_FBOBJC_MACOSX_COMPILER_FLAGS,
        fbobjc_macosx_configs_override = fbobjc_configs(
            STATIC_LIBRARY_MAC_CONFIG,
        ),
        macosx_frameworks_override = get_c2_fbobjc_frameworks(),
        preprocessor_flags = [
            # Use the internal pthreadpool impl for all Caffe2 targets on all
            # platforms but do not export the preprocessor flag downstream.
            "-DUSE_INTERNAL_PTHREADPOOL_IMPL",
        ],
        visibility = ["PUBLIC"],
        windows_preferred_linkage = "static" if is_arvr_mode() else None,
        xcode_public_headers_symlinks = True,
    )

def get_c2_aten_cpu_fbobjc_macosx_deps():
    return select({
        "DEFAULT": [],
        "ovr_config//os:macos-x86_64": ["fbsource//xplat/deeplearning/fbgemm:fbgemm"],
    }) if is_arvr_mode() else []

def build_cpukernel_avx2():
    return read_bool("caffe2", "build_cpukernel_avx2", not is_arvr_mode())

def get_c2_aten_cpu_fbobjc_macosx_platform_deps():
    return compose_platform_setting_list([
        {
            "cpu": "x86_64",
            "flags": [
                "fbsource//xplat/deeplearning/fbgemm:fbgemmAppleMac",
            ] + ([
                "fbsource//xplat/caffe2:cpukernel_avx2AppleMac",
            ] if build_cpukernel_avx2() else []),
            "os": "macosx",
        },
        {
            "cpu": "arm64",
            "flags": ["fbsource//xplat/third-party/XNNPACK:XNNPACKAppleMac"],
            "os": "macosx",
        },
    ])

def using_protobuf_v3():
    # Consider migrating this to `read_config("protobuf", "use_v3")`
    # The `is_fbcode_mode_mac()` clause was added rather than changing to `read_config` to minimize changes in behavior
    return is_arvr_mode() or is_fbcode_mode_mac()

def get_c2_protobuf_dep():
    return "fbsource//third-party/protobuf:libprotobuf" if using_protobuf_v3() else "fbsource//xplat/third-party/protobuf:fb-protobuf-lite"

def c2_cxx_library(fbobjc_compiler_flags = [], **kwargs):
    args = get_c2_default_cxx_args()
    args.update(kwargs)
    args.setdefault("platforms", (ANDROID, APPLE, CXX, WINDOWS))

    # Make sure we don't overwrite custom `fbobjc_compiler_flags`
    args["fbobjc_compiler_flags"] = args.pop("fbobjc_compiler_flags", []) + fbobjc_compiler_flags

    fb_xplat_cxx_library(
        labels = [
            "supermodule:android/default/caffe2",
            "supermodule:ios/default/public.caffe2",
        ],
        feature = Msgr_Mobile_ProductInfra,
        **args
    )

def c2_protobuf_rule(protos):
    cpps = []
    headers = {}
    raw_headers = {}
    for p in protos:
        proto = paths.basename(p)
        protocexe = "$(exe fbsource//third-party/protobuf:protoc-host)" if is_arvr_mode() else "$(location fbsource//xplat/third-party/protobuf:protoc.Windows)"
        protocmd_exe = "powershell.exe -file $(location fbsource//xplat/caffe2/scripts:proto)\\proto.ps1 -Protoc {} -Unprocessed $SRCDIR/{} -Processed $SRCDIR/{} -out $OUT -srcdir $SRCDIR".format(protocexe, p, proto)
        protocmd = ("cp $SRCDIR/{} $SRCDIR/{} && chmod +w $SRCDIR/{} && echo \"option optimize_for = LITE_RUNTIME;\" >> $SRCDIR/{} && ".format(p, proto, proto, proto) +
                    "cp $SRCDIR/caffe2/proto/caffe2.proto $SRCDIR/caffe2.proto && chmod +w $SRCDIR/caffe2.proto && echo \"option optimize_for = LITE_RUNTIME;\" >> $SRCDIR/caffe2.proto && " +
                    "sed -i -e 's/caffe2\\/proto\\/caffe2.proto/caffe2.proto/g' $SRCDIR/{} && ".format(proto) +
                    ("$(exe fbsource//third-party/protobuf:protoc-host) " if using_protobuf_v3() else "$(exe fbsource//xplat/third-party/protobuf:protoc) --osx $(location fbsource//xplat/third-party/protobuf:protoc.Darwin) --linux $(location fbsource//xplat/third-party/protobuf:protoc.Linux) ") +
                    "-I $SRCDIR --cpp_out=$OUT $SRCDIR/{}".format(proto))
        buck_genrule(
            name = proto,
            srcs = sorted(collections.uniq([p, "caffe2/proto/caffe2.proto"])),
            cmd_exe = protocmd_exe,
            bash = protocmd,
            out = ".",
        )
        (name, _) = paths.split_extension(proto)
        cpp = name + ".pb.cc"
        h = name + ".pb.h"
        buck_genrule(
            name = h,
            cmd_exe = "@powershell -Command \" & { " + "(Get-Content $(location :{})\\{}".format(proto, h) + ") -replace \\\"caffe2.pb.h\\\", \\\"caffe2/proto/caffe2.pb.h\\\" | Set-Content $OUT } \"",
            bash = "cp -f $(location :{})/{} $OUT  && ".format(proto, h) +
                   "sed -i -e 's/caffe2.pb.h/caffe2\\/proto\\/caffe2.pb.h/g' $OUT",
            out = h,
        )
        headers["caffe2/proto/" + h] = ":{}".format(h)
        raw_headers[h] = ":{}".format(h)
        buck_genrule(
            name = cpp,
            cmd_exe = "@powershell -Command copy $(location :{})/{} $OUT".format(proto, cpp),
            bash = "cp -f $(location :{})/{} $OUT".format(proto, cpp),
            out = cpp,
        )
        cpps.append(":{}".format(cpp))
    return (cpps, headers, raw_headers)

# C2 uses lite version of protobuf while torch/jit uses some method only exists
# in full protobuf. This is a temporary workaround to enable experiment build.
# DO NOT USE IT IN PRODUCTION BUILD!
def c2_full_protobuf_rule(protos):
    prefix = "full_"
    cpps = []
    headers = {}
    raw_headers = {}
    for p in protos:
        proto = paths.basename(p)
        protocexe = "$(exe fbsource//third-party/protobuf:protoc-host)" if is_arvr_mode() else "$(location fbsource//xplat/third-party/protobuf:protoc.Windows)"
        protocmd_exe = "powershell.exe -file $(location fbsource//xplat/caffe2/scripts:proto)\\proto.ps1 -Protoc {} -Unprocessed $SRCDIR/{} -Processed $SRCDIR/{} -out $OUT -srcdir $SRCDIR".format(protocexe, p, proto)
        protocmd = ("cp $SRCDIR/{} $SRCDIR/{} && ".format(p, proto) +
                    "cp $SRCDIR/caffe2/proto/caffe2.proto $SRCDIR/caffe2.proto && " +
                    "sed -i -e 's/caffe2\\/proto\\/caffe2.proto/caffe2.proto/g' $SRCDIR/{} && ".format(proto) +
                    ("$(exe fbsource//third-party/protobuf:protoc-host) " if using_protobuf_v3() else "$(exe fbsource//xplat/third-party/protobuf:protoc) --osx $(location fbsource//xplat/third-party/protobuf:protoc.Darwin) --linux $(location fbsource//xplat/third-party/protobuf:protoc.Linux) ") +
                    "-I $SRCDIR --cpp_out=$OUT $SRCDIR/{}".format(proto))
        buck_genrule(
            name = prefix + proto,
            srcs = sorted(collections.uniq([p, "caffe2/proto/caffe2.proto"])),
            cmd = protocmd,
            cmd_exe = protocmd_exe,
            out = ".",
        )
        (name, _) = paths.split_extension(proto)
        cpp = name + ".pb.cc"
        h = name + ".pb.h"
        buck_genrule(
            name = prefix + h,
            cmd_exe = "@powershell -Command \" & { " + "(Get-Content $(location :{})\\{}".format(prefix + proto, h) + ") -replace \\\"caffe2.pb.h\\\", \\\"caffe2/proto/caffe2.pb.h\\\" | Set-Content $OUT } \"",
            bash = "cp -f $(location :{})/{} $OUT  && ".format(prefix + proto, h) +
                   "sed -i -e 's/caffe2.pb.h/caffe2\\/proto\\/caffe2.pb.h/g' $OUT",
            out = h,
        )
        headers["caffe2/proto/" + h] = ":{}".format(prefix + h)
        raw_headers[h] = ":{}".format(prefix + h)
        buck_genrule(
            name = prefix + cpp,
            cmd_exe = "@powershell -Command copy $(location :{})/{} $OUT".format(prefix + proto, cpp),
            bash = "cp -f $(location :{})/{} $OUT".format(prefix + proto, cpp),
            out = cpp,
        )
        cpps.append(":{}".format(prefix + cpp))
    return (cpps, headers, raw_headers)

def libcaffe2_cxx_library(name, use_hptt, **kwargs):
    c2_cxx_library(
        name = name,
        exported_deps = [
            "fbsource//xplat/caffe2/c10:c10",
            get_c2_protobuf_dep(),
            ":caffe2_protobuf_headers",
            ":pthreadpool",
            ":common_core",
            ":caffe2_proto_types",
        ],
        compiler_flags = get_c2_xplat_compiler_flags() if use_hptt else get_c2_xplat_no_hptt_compiler_flags(),
        exported_preprocessor_flags = get_c2_xplat_preprocessor_flags() if use_hptt else get_c2_xplat_no_hptt_preprocessor_flags(),
        cxx_preprocessor_flags = C2_XPLAT_SERVER_PREPROCESSOR_FLAGS,
        fbandroid_exported_preprocessor_flags = get_c2_fbandroid_xplat_compiler_flags(),
        fbobjc_exported_preprocessor_flags = get_c2_fbobjc_xplat_compiler_flags(),
        # Hack to work around lack of platform_srcs support in Xcode project generation.
        macosx_extra_xcode_sources_override = [],
        link_whole = True,
        **kwargs
    )

def c2_operator_library(name, **kwargs):
    dict_defs.key_extend(
        kwargs,
        "deps",
        [
            "fbsource//xplat/folly:molly",
            "fbsource//third-party/glog:glog",
            ":caffe2",
        ] + ([":aten_cpu"] if get_c2_expose_op_to_c10() else []),
    )

    # NOTE: Currently operators can "depend" on other operators, which is used
    # so that loading one will implicitly load the dependencies.  So, make sure
    # that no `--as-needed` flags pulled in from dependencies cause these
    # operator deps to get dropped.
    linker_flags = [] if (read_config("caffe2", "link_as_needed", "0") == "1") else ["-Wl,--no-as-needed"]
    c2_cxx_library(
        name = name,
        soname = "lib" + name + ".$(ext)",
        fbandroid_compiler_flags = get_c2_default_cxx_args()["fbandroid_compiler_flags"] + ["-Os"],
        fbobjc_compiler_flags = get_c2_default_cxx_args()["fbobjc_compiler_flags"] + ["-Oz", "-DCOMPILING_FOR_MIN_SIZE=1"],
        link_whole = True,
        cxx_exported_linker_flags = linker_flags,
        fbandroid_exported_linker_flags = linker_flags,
        exported_deps = [
            ":caffe2",
        ],
        **kwargs
    )

def c2_genrule(genrule, genfiles, prefix = "", src_path = "", header_namespace = ""):
    headers = {}
    srcs = []
    for generated_filename in genfiles:
        buck_genrule(
            name = prefix + generated_filename,
            bash = "cp -f $(location :{})/{} $OUT".format(genrule, src_path + generated_filename),
            cmd_exe = "@powershell -Command copy $(location :{})/{} $OUT".format(genrule, src_path + generated_filename),
            out = generated_filename,
        )
        rule = ":{}{}".format(prefix, generated_filename)
        headers[header_namespace + generated_filename] = rule
        srcs.append(rule)
    return {"headers": headers, "srcs": srcs}
