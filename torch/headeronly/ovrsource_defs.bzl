load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library")

cpu_supported_platforms = [
    "ovr_config//os:android",
    "ovr_config//os:iphoneos",
    "ovr_config//os:linux-x86_64",
    "ovr_config//os:macos",
    "ovr_config//os:windows-x86_64",
    "ovr_config//runtime:arm64-linux-ubuntu-neon",
    "ovr_config//os:linux-arm64",
]

def define_torch_headeronly_ovrsource(name, is_mobile):
    if is_mobile:
        pp_flags = ["-DC10_MOBILE=1"]
    else:
        pp_flags = []

    oxx_static_library(
        name = name,
        srcs = []
        compatible_with = cpu_supported_platforms,
        compiler_flags = select({
            "DEFAULT": [],
        }),
        include_directories = [".."],
        preprocessor_flags = [],
        fbobjc_compiler_flags = [],
        public_include_directories = [".."],
        public_preprocessor_flags = pp_flags,
        public_raw_headers = native.glob([
            "macros/*.h",
        ]),
        reexport_all_header_dependencies = False,
        visibility = [
            "//xplat/caffe2/torch/headeronly:torch_headeronly",
        ],
        deps = select({
            "DEFAULT": [],
        }),
    )

def define_ovrsource_targets():
    oxx_static_library(
        name = "torch_headeronly_ovrsource",
        compatible_with = cpu_supported_platforms,
        exported_deps = select({
            "DEFAULT": [":torch_headeronly_full_ovrsource"],
            "ovr_config//os:android": [":torch_headeronly_mobile_ovrsource"],
            "ovr_config//os:iphoneos": [":torch_headeronly_mobile_ovrsource"],
        }),
        visibility = ["PUBLIC"],
    )

    """
    Most users should use torch_headeronly_ovrsource, not these targets directly.
    """
    define_torch_headeronly_ovrsource("torch_headeronly_mobile_ovrsource", True)
    define_torch_headeronly_ovrsource("torch_headeronly_full_ovrsource", False)
