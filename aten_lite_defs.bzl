"""Macros for building a selective ATen CUDA registration library composed on
top of a CPU-side `pt_operator_registry` / `get_pt_ops_deps` library.

The stock `pt_operator_library` selective-build path in
`xplat/caffe2:pt_defs.bzl` hard-codes `enabled_backends = USED_PT_BACKENDS =
[CPU, QuantizedCPU, SparseCPU]` and only emits CPU-side registration. The
macros here drive `gen_aten_files` with a broader backend list and link the
generated `Register*CUDA_*.cpp` + `aten_ufunc_generated_cuda_sources` outputs
into a `link_whole=True` library, paired with the kernels-only
`ATen_cuda_kernels_only_ovrsource` sibling for the actual CUDA kernel
implementations.

Two public macros:

  * `define_aten_lite_from_yaml(name, backends, op_selection_yaml_path, ...)`
    consumes an existing `selected_operators.yaml` (typically the
    `<name>_pt_oplist` genrule output from `pt_operator_registry` /
    `get_pt_ops_deps`).

  * `define_aten_lite(name, backends, oplist, ...)` takes the operator list
    inline as `["aten::cat", "aten::matmul", ...]`, emits a minimal
    `selected_operators.yaml` via a private genrule, and forwards.

Example (inline oplist):

    load("@fbsource//xplat/caffe2:aten_lite_defs.bzl", "define_aten_lite")

    define_aten_lite(
        name = "my_aten_cuda_lite",
        backends = ["CPU", "QuantizedCPU", "SparseCPU", "CUDA"],
        oplist = [
            "aten::cat",
            "aten::matmul",
        ],
        visibility = ["PUBLIC"],
    )

Example (existing `selected_operators.yaml`):

    load("@fbsource//xplat/caffe2:aten_lite_defs.bzl", "define_aten_lite_from_yaml")
    load("@fbsource//xplat/caffe2:pt_defs.bzl", "get_pt_ops_deps")
    load("@fbsource//xplat/caffe2:pt_ops.bzl", "pt_operator_library")

    pt_operator_library(
        name = "my_ops",
        ops = ["aten::cat", "aten::matmul"],
    )

    get_pt_ops_deps("my_selective_ops", [":my_ops"])

    define_aten_lite_from_yaml(
        name = "my_aten_cuda_lite",
        backends = ["CPU", "QuantizedCPU", "SparseCPU", "CUDA"],
        op_selection_yaml_path = "$(location :my_selective_ops_pt_oplist[selected_operators.yaml])",
        visibility = ["PUBLIC"],
    )
"""

load("@bazel_skylib//lib:shell.bzl", "shell")
load("@fbsource//tools/build_defs:fb_xplat_genrule.bzl", "fb_xplat_genrule")
load("@fbsource//xplat/caffe2:buckbuild.bzl", "gen_aten_files")
load("@fbsource//xplat/caffe2:ufunc_defs.bzl", "aten_ufunc_generated_cuda_sources")
load("@fbsource//xplat/caffe2/c10:ovrsource_defs.bzl", "cuda_supported_platforms")
load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library")

_EMIT_YAML_BIN = "//xplat/caffe2/tools:emit_selected_operators_yaml"

_RECOGNIZED_CPU_BACKENDS = [
    "CPU",
    "QuantizedCPU",
    "SparseCPU",
]

# Mirrors the four CUDA-side `Register*CUDA_0.cpp` entries in
# `ATen_cuda_lib_ovrsource`. Passing "CUDA" auto-expands to this full set.
_CUDA_FAMILY_BACKENDS = [
    "CUDA",
    "SparseCUDA",
    "SparseCsrCUDA",
    "QuantizedCUDA",
]

_RECOGNIZED_BACKENDS = _RECOGNIZED_CPU_BACKENDS + _CUDA_FAMILY_BACKENDS

def _expand_backends(backends):
    expanded = []
    seen = {}
    has_cuda_family = False
    for backend in backends:
        if backend in _CUDA_FAMILY_BACKENDS:
            has_cuda_family = True
            continue
        if backend in seen:
            continue
        seen[backend] = True
        expanded.append(backend)
    if has_cuda_family:
        for backend in _CUDA_FAMILY_BACKENDS:
            if backend in seen:
                continue
            seen[backend] = True
            expanded.append(backend)
    return expanded

def define_aten_lite_from_yaml(
        name,
        backends,
        op_selection_yaml_path,
        visibility = [],
        force_schema_registration = True):
    """Build a selective ATen CUDA registration library.

    Emits a `gen_aten_files` genrule, a private CUDA-side headers target
    (`CUDAFunctions.h`, `CUDAFunctions_inl.h`), and a `link_whole=True`
    `oxx_static_library` named `name` containing the generated
    `Register<CudaBackend>_0.cpp` sources plus `aten_ufunc_generated_cuda_sources`
    outputs. CUDA kernel implementations are pulled in via
    `xplat/caffe2:ATen_cuda_kernels_only_ovrsource`.

    Args:
        name: Target name for the resulting `link_whole=True` registration
            library.
        backends: Must include at least one CUDA-family backend ("CUDA",
            "SparseCUDA", "SparseCsrCUDA", or "QuantizedCUDA") and at least
            one CPU-side backend ("CPU", "QuantizedCPU", or "SparseCPU").
            "CUDA" auto-expands to the full CUDA family. The CPU-side
            `Register*.cpp` files are NOT compiled into this library
            (they come from `pt_operator_registry`); CPU backends still
            need to appear here because the generated CUDA `Register*.cpp`
            files reference CPU-side dispatch entries that codegen only
            emits when CPU is in the backend whitelist.
        op_selection_yaml_path: A `$(location ...)` reference resolving to
            a `selected_operators.yaml` accepted by `gen_aten_files`'s
            `op_selection_yaml_path` extra flag. Typically the
            `<name>_pt_oplist` genrule output from `pt_operator_registry` /
            `get_pt_ops_deps`.
        visibility: Visibility for the registration library only. The
            internal genrule and headers target stay package-private.
        force_schema_registration: Forwarded to `gen_aten_files`. Defaults
            to True to match `pt_operator_library`.
    """
    for backend in backends:
        if backend not in _RECOGNIZED_BACKENDS:
            fail(
                "define_aten_lite_from_yaml({name}): unrecognized backend {backend!r}. ".format(
                    name = name,
                    backend = backend,
                ) +
                "Must be one of: {recognized}.".format(
                    recognized = ", ".join(_RECOGNIZED_BACKENDS),
                ),
            )

    expanded_backends = _expand_backends(backends)
    has_cuda = any([b in _CUDA_FAMILY_BACKENDS for b in expanded_backends])
    has_cpu = any([b in _RECOGNIZED_CPU_BACKENDS for b in expanded_backends])

    if not has_cuda:
        fail(
            "define_aten_lite_from_yaml({name}): `backends` must include a CUDA-family ".format(name = name) +
            "backend (one of {family}). ".format(family = ", ".join(_CUDA_FAMILY_BACKENDS)) +
            "For CPU-only selective op registration, use " +
            "`pt_operator_registry` / `get_pt_ops_deps` from " +
            "`xplat/caffe2:pt_defs.bzl` instead.",
        )

    if not has_cpu:
        fail(
            "define_aten_lite_from_yaml({name}): `backends` must include at least one ".format(name = name) +
            "CPU-side backend (one of {cpu}). ".format(cpu = ", ".join(_RECOGNIZED_CPU_BACKENDS)) +
            "Typical usage: `backends = [\"CPU\", \"QuantizedCPU\", \"SparseCPU\", \"CUDA\"]`.",
        )

    package_visibility = ["//{}/...".format(native.package_name())]

    aten_genrule = name + "_gen"
    gen_aten_files(
        name = aten_genrule,
        extra_flags = {
            "enabled_backends": expanded_backends,
            "force_schema_registration": force_schema_registration,
            "op_selection_yaml_path": op_selection_yaml_path,
        },
        compatible_with = cuda_supported_platforms,
        visibility = package_visibility,
    )

    cuda_headers_target = name + "_cuda_headers"
    oxx_static_library(
        name = cuda_headers_target,
        compatible_with = cuda_supported_platforms,
        header_namespace = "ATen",
        public_generated_headers = {
            "CUDAFunctions.h": ":{}[CUDAFunctions.h]".format(aten_genrule),
            "CUDAFunctions_inl.h": ":{}[CUDAFunctions_inl.h]".format(aten_genrule),
        },
        visibility = package_visibility,
    )

    srcs = [
        ":{}[Register{}_0.cpp]".format(aten_genrule, b)
        for b in _CUDA_FAMILY_BACKENDS
        if b in expanded_backends
    ] + aten_ufunc_generated_cuda_sources(
        ":{}[{{}}]".format(aten_genrule),
    )

    public_deps = [
        "//xplat/caffe2:torch_mobile_core",
    ] + select({
        "DEFAULT": [],
        "ovr_config//cuda:has_cuda": [
            "//xplat/caffe2:ATen_cuda_headers_ovrsource",
            "//xplat/caffe2:ATen_cuda_kernels_only_ovrsource",
            "//xplat/caffe2/c10:c10_cuda_ovrsource",
            ":" + cuda_headers_target,
            "//third-party/cuda:libcudart",
            "//third-party/cuda:libcublas",
            "//third-party/cuda:libcusolver",
            "//third-party/cuda:libcusparse",
        ],
    })

    cuda_private_deps = select({
        "DEFAULT": [],
        "ovr_config//cuda:has_cuda": [
            "//third-party/cuda:libcuda",
            "//third-party/cuda:libcufft",
            "//third-party/cuda:libnvrtc",
            "//third-party/cuda:libthrust",
            "//third-party/cudnn:libcudnn",
        ],
    })

    # @lint-ignore BUCKLINT link_whole
    oxx_static_library(
        name = name,
        srcs = srcs,
        compatible_with = cuda_supported_platforms,
        link_whole = True,
        public_deps = public_deps,
        deps = cuda_private_deps,
        visibility = visibility,
    )

def _emit_selected_operators_yaml_genrule(name, oplist):
    # Use a Python helper rather than shell echo/printf. cmd.exe expands
    # `echo off` / `echo on` to a YAML scalar `false` / `true`, breaking
    # downstream codegen with `'bool' object has no attribute 'keys'`.
    quoted_ops = " ".join([shell.quote(op) for op in oplist])
    cmd = "$(exe {bin}) --output $OUT/selected_operators.yaml {ops}".format(
        bin = _EMIT_YAML_BIN,
        ops = quoted_ops,
    )

    fb_xplat_genrule(
        name = name,
        outs = {"selected_operators.yaml": ["selected_operators.yaml"]},
        default_outs = ["."],
        cmd = cmd,
        visibility = ["//{}/...".format(native.package_name())],
    )

def define_aten_lite(
        name,
        backends,
        oplist,
        visibility = [],
        force_schema_registration = True):
    """Inline-oplist wrapper around `define_aten_lite_from_yaml`.

    Writes a minimal `selected_operators.yaml` for `oplist` via a private
    genrule and forwards to `define_aten_lite_from_yaml`. Bare operator
    names (no `.OverloadName`) select every overload; no closure expansion
    is performed (unlike `pt_operator_library`), so the caller must list
    every operator the selective codegen needs to see.
    """
    yaml_genrule = name + "_yaml"
    _emit_selected_operators_yaml_genrule(
        name = yaml_genrule,
        oplist = oplist,
    )

    define_aten_lite_from_yaml(
        name = name,
        backends = backends,
        op_selection_yaml_path = "$(location :{}[selected_operators.yaml])".format(yaml_genrule),
        visibility = visibility,
        force_schema_registration = force_schema_registration,
    )
