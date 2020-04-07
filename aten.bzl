load("@rules_cc//cc:defs.bzl", "cc_library")

CPU_CAPABILITY_NAMES = ["DEFAULT", "AVX", "AVX2"]
PREFIX = "aten/src/ATen/native/"

def intern_build_aten_ops(copts, deps):
    for cpu_capability in CPU_CAPABILITY_NAMES:
        srcs = []
        for impl in native.glob(
            [
                PREFIX + "cpu/*.cpp",
                PREFIX + "quantized/cpu/kernels/*.cpp",
            ]):
            name = impl.replace(PREFIX, "")
            out = PREFIX + name + "." + cpu_capability + ".cpp"
            native.genrule(
                name = name + "_" + cpu_capability + "_cp",
                srcs = [impl],
                outs = [out],
                cmd = "cp $< $@",
            )
            srcs.append(out)

        cc_library(
            name = "ATen_CPU_" + cpu_capability,
            srcs = srcs,
            copts = copts + [
                "-DCPU_CAPABILITY=" + cpu_capability,
                "-DCPU_CAPABILITY_" + cpu_capability,
            ],
            deps = deps,
        )
    cc_library(
        name = "ATen_CPU",
        srcs = ["ATen_CPU_" + cpu_capability for cpu_capability in CPU_CAPABILITY_NAMES],
        linkstatic = 1,
    )
