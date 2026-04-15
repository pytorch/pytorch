# PyTorch-maintained BUILD file for cpuinfo.
# The upstream BUILD.bazel requires rules_cc 0.2.12+ / Bazel 7+,
# so we keep a stripped-down copy that works with Bazel 6.5.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

C99OPTS = [
    "-std=gnu99",  # gnu99, not c99, because dprintf is used
    "-Wno-vla",
    "-D_GNU_SOURCE=1",  # to use CPU_SETSIZE
    "-DCPUINFO_INTERNAL=",
    "-DCPUINFO_PRIVATE=",
]

COMMON_SRCS = [
    "src/api.c",
    "src/cache.c",
    "src/init.c",
    "src/log.c",
]

X86_SRCS = [
    "src/x86/cache/descriptor.c",
    "src/x86/cache/deterministic.c",
    "src/x86/cache/init.c",
    "src/x86/info.c",
    "src/x86/init.c",
    "src/x86/isa.c",
    "src/x86/name.c",
    "src/x86/topology.c",
    "src/x86/uarch.c",
    "src/x86/vendor.c",
]

ARM_SRCS = [
    "src/arm/cache.c",
    "src/arm/uarch.c",
]

RISCV_SRCS = [
    "src/riscv/uarch.c",
]

LINUX_SRCS = [
    "src/linux/cpulist.c",
    "src/linux/multiline.c",
    "src/linux/processors.c",
    "src/linux/smallfile.c",
]

MACH_SRCS = [
    "src/mach/topology.c",
]

LINUX_X86_SRCS = [
    "src/x86/linux/cpuinfo.c",
    "src/x86/linux/init.c",
]

LINUX_ARM_SRCS = [
    "src/arm/linux/chipset.c",
    "src/arm/linux/clusters.c",
    "src/arm/linux/cpuinfo.c",
    "src/arm/linux/hwcap.c",
    "src/arm/linux/init.c",
    "src/arm/linux/midr.c",
]

LINUX_ARM32_SRCS = LINUX_ARM_SRCS + ["src/arm/linux/aarch32-isa.c"]

LINUX_ARM64_SRCS = LINUX_ARM_SRCS + ["src/arm/linux/aarch64-isa.c"]

LINUX_RISCV_SRCS = [
    "src/riscv/linux/init.c",
    "src/riscv/linux/riscv-isa.c",
    "src/riscv/linux/riscv-hw.c",
]

MACH_X86_SRCS = [
    "src/x86/mach/init.c",
]

MACH_ARM_SRCS = [
    "src/arm/mach/init.c",
]

WINDOWS_X86_SRCS = [
    "src/x86/windows/init.c",
]

cc_library(
    name = "cpuinfo_impl",
    srcs = select({
        ":linux_x86_64": COMMON_SRCS + X86_SRCS + LINUX_SRCS + LINUX_X86_SRCS,
        ":linux_arm": COMMON_SRCS + ARM_SRCS + LINUX_SRCS + LINUX_ARM32_SRCS,
        ":linux_armhf": COMMON_SRCS + ARM_SRCS + LINUX_SRCS + LINUX_ARM32_SRCS,
        ":linux_armv7a": COMMON_SRCS + ARM_SRCS + LINUX_SRCS + LINUX_ARM32_SRCS,
        ":linux_armeabi": COMMON_SRCS + ARM_SRCS + LINUX_SRCS + LINUX_ARM32_SRCS,
        ":linux_aarch64": COMMON_SRCS + ARM_SRCS + LINUX_SRCS + LINUX_ARM64_SRCS,
        ":linux_mips64": COMMON_SRCS + LINUX_SRCS,
        ":linux_ppc64le": COMMON_SRCS + LINUX_SRCS,
        ":linux_riscv32": COMMON_SRCS + RISCV_SRCS + LINUX_SRCS + LINUX_RISCV_SRCS,
        ":linux_riscv64": COMMON_SRCS + RISCV_SRCS + LINUX_SRCS + LINUX_RISCV_SRCS,
        ":linux_s390x": COMMON_SRCS + LINUX_SRCS,
        ":macos_x86_64": COMMON_SRCS + X86_SRCS + MACH_SRCS + MACH_X86_SRCS,
        ":macos_x86_64_legacy": COMMON_SRCS + X86_SRCS + MACH_SRCS + MACH_X86_SRCS,
        ":macos_arm64": COMMON_SRCS + MACH_SRCS + MACH_ARM_SRCS,
        ":windows_x86_64": COMMON_SRCS + X86_SRCS + WINDOWS_X86_SRCS,
    }),
    copts = select({
        ":windows_x86_64": [],
        "//conditions:default": C99OPTS,
    }) + [
        "-Iexternal/cpuinfo/include",
        "-Iexternal/cpuinfo/src",
        "-DCPUINFO_LOG_LEVEL=2",
    ],
    includes = [
        "include",
        "src",
    ],
    linkstatic = select({
        # https://github.com/bazelbuild/bazel/issues/11552
        ":macos_x86_64": False,
        ":macos_x86_64_legacy": False,
        "//conditions:default": True,
    }),
    textual_hdrs = [
        "include/cpuinfo.h",
        "src/linux/api.h",
        "src/mach/api.h",
        "src/cpuinfo/common.h",
        "src/cpuinfo/internal-api.h",
        "src/cpuinfo/log.h",
        "src/cpuinfo/utils.h",
        "src/x86/api.h",
        "src/x86/cpuid.h",
        "src/x86/linux/api.h",
        "src/arm/linux/api.h",
        "src/arm/linux/cp.h",
        "src/arm/api.h",
        "src/arm/midr.h",
        "src/riscv/api.h",
        "src/riscv/linux/api.h",
    ],
)

cc_library(
    name = "cpuinfo",
    hdrs = [
        "include/cpuinfo.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":cpuinfo_impl",
    ],
)

cc_library(
    name = "cpuinfo_with_unstripped_include_path",
    hdrs = [
        "include/cpuinfo.h",
    ],
    deps = [
        ":cpuinfo_impl",
    ],
)

############################# Build configurations #############################

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
)

config_setting(
    name = "linux_arm",
    values = {"cpu": "arm"},
)

config_setting(
    name = "linux_armhf",
    values = {"cpu": "armhf"},
)

config_setting(
    name = "linux_armv7a",
    values = {"cpu": "armv7a"},
)

config_setting(
    name = "linux_armeabi",
    values = {"cpu": "armeabi"},
)

config_setting(
    name = "linux_aarch64",
    values = {"cpu": "aarch64"},
)

config_setting(
    name = "linux_mips64",
    values = {"cpu": "mips64"},
)

config_setting(
    name = "linux_ppc64le",
    values = {"cpu": "ppc"},
)

config_setting(
    name = "linux_riscv32",
    values = {"cpu": "riscv32"},
)

config_setting(
    name = "linux_riscv64",
    values = {"cpu": "riscv64"},
)

config_setting(
    name = "linux_s390x",
    values = {"cpu": "s390x"},
)

config_setting(
    name = "macos_x86_64_legacy",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
)

config_setting(
    name = "macos_x86_64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_x86_64",
    },
)

config_setting(
    name = "macos_arm64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64",
    },
)

config_setting(
    name = "windows_x86_64",
    values = {"cpu": "x64_windows"},
)
