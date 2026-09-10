"""Toolchain abstraction for native-AOT kernel export and launcher codegen.

One ``Toolchain`` subclass per DSL/compiler. Everything per-kind lives
here -- how a builder dict is validated and compiled into artifacts, what
those artifacts are (extensions, link inputs), and how the C++ launcher
marshals at::Tensor arguments into the kernel's ABI -- so adding a DSL
means adding one class and registering it, with export.py, gen_aot_lib.py
and the CMake project untouched.

The cross-toolchain contract, in order of flow:

  1. An op's builder module (package-imported by export.py) exposes
     ``build(spec) -> dict``. The dict's ``kind`` selects the toolchain;
     ``prefix`` names the artifacts; ``REQUIRED_BUILD_KEYS`` lists what
     else the toolchain needs (validated up front so a bad builder fails
     at export with a message, not a KeyError mid-compile).
  2. ``export(build_result, out_dir)`` runs the compiler and writes
     ``<prefix>.<artifact_exts>``. It returns the marshalling metadata to
     persist in the ``<prefix>.json`` sidecar -- the ONLY channel from
     export-time knowledge to launcher codegen.
  3. ``gen_launcher(sidecar)`` emits C++ with the toolchain-independent
     signature every declaration's ``cpp_launch()`` programs against:

         void launch_<prefix>(const at::Tensor&..., <scalars>..., c10::Stream)

     Everything above the launcher (guard chain, cond, DispatchStub
     registration) is toolchain-blind. The stream crosses that boundary
     as a device-agnostic c10::Stream, so the shared contract carries no
     CUDA type; each launcher body narrows it to the raw handle its own
     C ABI needs (cute_dsl_*_wrapper, the Triton entry point and
     cuLaunchKernel all take cudaStream_t/CUstream). Declarations pass
     at::cuda::getCurrentCUDAStream(), which converts implicitly.

Properties consumed by the driver scripts:
  * ``artifact_exts``: extensions written next to the sidecar; used to
    spot artifacts left with no sidecar (export._check_no_orphan_artifacts).
    The idempotency skip keys on the sidecar itself, not on these.
  * ``link_source_globs``: artifact patterns the CMake project must
    compile or link (kept in sync with the embedded-link block in
    caffe2/CMakeLists.txt, which cannot import this file; see the
    assertion in the tests).
  * ``launcher_includes``: per-kind includes for the generated .cpp.
  * ``kernel_includes(sidecar)``: per-kernel includes for that same file,
    for toolchains whose export writes a header (CuTeDSL's ABI struct).
  * ``NARROWS_SHAPES_TO_INT32``: the exported ABI takes i32 extents, so
    gen_aot_lib emits a stub gate declining dims past INT32_MAX.
  * ``ARCH_ENV_VAR``: env var this kind reads when no arch is passed
    explicitly; export._effective_arch resolves it so the sidecar records
    the arch actually compiled for and a run that changes only that
    variable is not skipped as already exported.

Export runs as stage 2 of the two-stage build (build torch -> build
the AOT lib), so torch is always importable during export.
"""

from __future__ import annotations


class Toolchain:
    kind: str = ""
    artifact_exts: tuple[str, ...] = ()
    link_source_globs: tuple[str, ...] = ()
    launcher_includes: tuple[str, ...] = ()

    # Torch build backends this kind can emit kernels for, as the names
    # torch reports: "cuda" or "rocm". A build whose backend is not listed
    # never asks for this toolchain, so its runtime being absent there is
    # EXPECTED, not a skip worth reporting -- that is what keeps a ROCm
    # build from complaining about missing CuTeDSL/Triton wheels while a
    # CUDA build still fails loudly if a declared kernel cannot be built.
    # Declared per-kind rather than inferred so a future ROCm DSL (FlyDSL)
    # is a new class with BACKENDS = ("rocm",) and no gate to rewrite.
    BACKENDS: tuple[str, ...] = ("cuda",)

    # Importable modules this kind needs to COMPILE a kernel. Absence is
    # FATAL once a declaration targeting this build's backend reaches
    # export: its kernels were asked for, and exporting only some of them
    # ships a wheel that silently underperforms. Build without the DSL
    # wheels via TORCH_NATIVE_AOT=0 instead (see build_stage2.should_run
    # and test_missing_runtime_is_fatal_not_skipped). Nothing at RUNTIME
    # needs these -- the exported artifacts are self-contained.
    REQUIRED_RUNTIMES: tuple[str, ...] = ()

    # True when this kind's exported ABI carries int32_t shape slots, so a
    # dim past INT32_MAX cannot be passed and the generated stub must
    # decline the call (gen_aot_lib's _int32_size_gate). A property of the
    # exported ABI, not of aten, so it lives per-kind: Triton takes its
    # scalar widths from the kernel's own signature and needs no such gate.
    NARROWS_SHAPES_TO_INT32: bool = False

    # Env var this kind falls back to when no explicit arch is given. The
    # sidecar records the arch it resolves to, so a run that changes only
    # this variable is not mistaken for one that already exported (see
    # export._effective_arch).
    ARCH_ENV_VAR: str | None = None

    REQUIRED_BUILD_KEYS: tuple[str, ...] = ()

    @classmethod
    def missing_runtimes(cls) -> list[str]:
        import importlib.util

        return [m for m in cls.REQUIRED_RUNTIMES if importlib.util.find_spec(m) is None]

    @classmethod
    def serves_backend(cls, backend: str) -> bool:
        return backend in cls.BACKENDS

    def validate_build_result(self, b: dict) -> None:
        missing = [k for k in ("prefix", *self.REQUIRED_BUILD_KEYS) if k not in b]
        if missing:
            raise RuntimeError(
                f"{self.kind} builder result for "
                f"{b.get('prefix', '<unnamed>')} is missing keys: {missing}"
            )

    def export(self, b: dict, out_dir: str, arch: str | None = None) -> dict:
        """Compile one spec point; return sidecar marshalling metadata.

        ``arch`` is an sm string ("sm_90a") or None for detect-from-
        device. With an explicit arch no toolchain touches the CUDA
        driver, so export runs on GPU-less machines, and the arch is
        per-compile state rather than per-process: CuTeDSL passes
        --gpu-arch, Triton kinds install a fixed GPUTarget driver. One
        process may therefore export for several arches."""
        raise NotImplementedError

    def gen_launcher(self, sidecar: dict) -> str:
        """Emit the launch_<prefix>() helper for one sidecar."""
        raise NotImplementedError

    def kernel_includes(self, sidecar: dict) -> list[str]:
        """Per-KERNEL includes for the generated .cpp, e.g. the ABI header
        export wrote. launcher_includes is the per-KIND counterpart, and
        most toolchains need only that."""
        return []


class CuteDslToolchain(Toolchain):
    """cute.compile + export_to_c: a .o kernel object plus a header of
    per-tensor ABI structs ({data, dynamic shape/stride slots}); module
    (cubin) load is explicit and eager across devices."""

    kind = "cutedsl"
    artifact_exts = (".o", ".h")
    link_source_globs = ("*/*.o",)
    launcher_includes = ()  # per-kernel header, included by prefix below

    # export_to_c emits `int32_t dynamic_shapes[]`, so the generated stub
    # must decline dims that do not fit (see gen_aot_lib._int32_size_gate).
    NARROWS_SHAPES_TO_INT32 = True

    ARCH_ENV_VAR = "CUTE_DSL_ARCH"

    # tvm_ffi: the JIT wrappers pass --enable-tvm-ffi, and cutlass imports
    # it during compile even though the exported ABI does not use it.
    REQUIRED_RUNTIMES = ("cutlass", "tvm_ffi")
    REQUIRED_BUILD_KEYS = ("fn", "fake_args", "tensor_args")

    # Rendered into the generated file's anonymous namespace, so the module
    # handle, the once_flag and launch_ itself all get internal linkage --
    # nothing here is part of libtorch_cuda's exported surface.
    LAUNCHER_TMPL = """\
{prefix}_Kernel_Module_t {prefix}_module;
c10::once_flag {prefix}_loaded;

void launch_{prefix}({tparams}, c10::Stream stream) {{
  c10::call_once({prefix}_loaded, [] {{ {prefix}_Kernel_Module_Load(&{prefix}_module); }});
{fills}
  int32_t rc = cute_dsl_{prefix}_wrapper(&{prefix}_module, {call_args},
                                         c10::cuda::CUDAStream(stream).stream());
  TORCH_CHECK(rc == 0, "{prefix} launch failed with code ", rc);
}}
"""

    _warmed_up = False

    @classmethod
    def _warm_up_exporter(cls) -> None:
        """Build one JIT engine in this process so export_to_c works for
        ANY --gpu-arch.

        export_to_c ultimately calls export_module_to_bytes(...,
        enable_pic=True), which needs the LLVM machinery that only gets
        initialized when the DSL creates a JIT engine. dsl.py creates one
        when `num_kernels == 0 or compile_gpu_arch == envar.arch`, so
        without this a cross-arch export is the FIRST compile in a fresh
        process and dies with "Failed to dump object file with PIC
        relocation" -- while the very same call succeeds if any
        engine-creating compile ran before it.

        A kernel-free @cute.jit takes the num_kernels == 0 branch, so it
        initializes the engine for any target: ~0.12s, once per process,
        no CUDA device needed. That is what lets one worker export for
        any mix of arches.
        """
        if cls._warmed_up:
            return
        # Defined in a helper module, not here: this file has `from
        # __future__ import annotations`, which turns the jit function's
        # parameter annotation into a string the DSL cannot resolve
        # ("NameError: name 'Float32' is not defined").
        from tools.native_aot.cutedsl_warmup import warm_up

        warm_up()
        cls._warmed_up = True

    def export(self, b: dict, out_dir: str, arch: str | None = None) -> dict:
        import cutlass.cute as cute

        # Optional compile options (e.g. --enable-assertions). Must match
        # what the op's JIT wrapper passes (minus --enable-tvm-ffi, which
        # would change the exported ABI) or the two routes' SASS diverges.
        opts = b.get("options")
        if arch:
            # --gpu-arch is authoritative: dsl.py prefers
            # compile_options.gpu_arch over the CUTE_DSL_ARCH env var, so
            # one process can export for several arches.
            opts = f"{opts} --gpu-arch {arch}" if opts else f"--gpu-arch {arch}"
            self._warm_up_exporter()
        compiled = cute.compile(
            b["fn"], *b["fake_args"], **({"options": opts} if opts else {})
        )
        compiled.export_to_c(
            file_path=out_dir, file_name=b["prefix"], function_prefix=b["prefix"]
        )
        sidecar = {"tensor_args": b["tensor_args"]}
        if b.get("scalar_args"):
            sidecar["scalar_args"] = b["scalar_args"]
        return sidecar

    def kernel_includes(self, sidecar: dict) -> list[str]:
        return [f'#include "{sidecar["prefix"]}.h"']

    def gen_launcher(self, sidecar: dict) -> str:
        prefix = sidecar["prefix"]
        targs = sidecar["tensor_args"]
        sargs = sidecar.get("scalar_args", [])
        fills = []
        for a in targs:
            n = a["name"]
            fills.append(f"  {prefix}_Tensor_{n}_t {n}_s;")
            if a.get("read_only"):
                # const_data_ptr: a mutable data_ptr() would materialize
                # copy-on-write inputs (PAIN_POINTS P15). The ABI struct
                # field is void*, hence the const_cast; the kernel only
                # reads through it (declaration's read_only promise).
                fills.append(f"  {n}_s.data = const_cast<void*>({n}.const_data_ptr());")
            else:
                fills.append(f"  {n}_s.data = {n}.mutable_data_ptr();")
            for slot, dim in enumerate(a.get("dynamic_sizes", [])):
                fills.append(
                    f"  {n}_s.dynamic_shapes[{slot}] = static_cast<int32_t>({n}.size({dim}));"
                )
            for slot, dim in enumerate(a.get("dynamic_strides", [])):
                fills.append(f"  {n}_s.dynamic_strides[{slot}] = {n}.stride({dim});")
        params = [f"const at::Tensor& {a['name']}" for a in targs]
        params += [f"{a['ctype']} {a['name']}" for a in sargs]
        # Wrapper argument order matches the exported signature: tensor
        # structs first, then scalars (by value), then the stream.
        call_args = [f"&{a['name']}_s" for a in targs] + [a["name"] for a in sargs]
        return self.LAUNCHER_TMPL.format(
            prefix=prefix,
            tparams=", ".join(params),
            fills="\n".join(fills),
            call_args=", ".join(call_args),
        )


TOOLCHAINS: dict[str, Toolchain] = {tc.kind: tc for tc in (CuteDslToolchain(),)}


def get_toolchain(kind: str) -> Toolchain:
    if kind not in TOOLCHAINS:
        raise RuntimeError(
            f"unknown toolchain kind {kind!r}; known: {sorted(TOOLCHAINS)}"
        )
    return TOOLCHAINS[kind]


def for_backend(backend: str) -> dict[str, Toolchain]:
    """The toolchains that can emit kernels for this torch build backend
    ("cuda" or "rocm"), per each kind's BACKENDS."""
    return {k: tc for k, tc in TOOLCHAINS.items() if tc.serves_backend(backend)}
