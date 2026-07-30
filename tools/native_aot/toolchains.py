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
     signature every manifest ``body`` programs against:

         void launch_<prefix>(const at::Tensor&..., <scalars>..., cudaStream_t)

     Everything above the launcher (guard chain, cond, DispatchStub
     registration) is toolchain-blind.

Properties consumed by the driver scripts:
  * ``artifact_exts``: extensions written next to the sidecar; the first
    is probed for idempotency-skip.
  * ``link_source_globs``: artifact patterns the CMake project must
    compile or link (kept in sync with the embedded-link block in
    caffe2/CMakeLists.txt, which cannot import this file; see the
    assertion in the tests).
  * ``launcher_includes``: per-kind includes for the generated .cpp.

Export runs as stage 2 of the two-stage build (build torch -> build
the AOT lib), so torch is always importable during export.
"""

from __future__ import annotations


class Toolchain:
    kind: str = ""
    artifact_exts: tuple[str, ...] = ()
    link_source_globs: tuple[str, ...] = ()
    launcher_includes: tuple[str, ...] = ()

    REQUIRED_BUILD_KEYS: tuple[str, ...] = ()

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
        driver, so export runs on GPU-less machines (CuTeDSL reads
        CUTE_DSL_ARCH -- set by the export tool BEFORE cutlass imports,
        it is cached at first read; Triton kinds build an explicit
        GPUTarget)."""
        raise NotImplementedError

    @staticmethod
    def _sm_number(arch: str) -> int:
        # "sm_90a" / "sm_100" -> 90 / 100 (Triton GPUTarget arch int).
        import re

        m = re.fullmatch(r"sm_(\d+)a?", arch)
        if not m:
            raise ValueError(f"arch must look like sm_90a, got {arch!r}")
        return int(m.group(1))

    @classmethod
    def _activate_triton_target(cls, arch: str):
        """Point triton's active driver at an explicit GPUTarget so
        compilation never queries a device (create_binder calls
        driver.active.get_current_target() unconditionally; the stock
        CudaDriver answers it via torch.cuda.current_device, which
        initializes CUDA and throws on GPU-less machines). Returns the
        GPUTarget. Process-wide, matching CUTE_DSL_ARCH's semantics on
        the CuTeDSL side; export workers are per-arch processes."""
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.backends.nvidia.driver import CudaDriver

        target = GPUTarget("cuda", cls._sm_number(arch), 32)

        class _FixedTargetDriver(CudaDriver):
            def get_current_target(self):
                return target

        triton.runtime.driver.set_active(_FixedTargetDriver())
        return target

    def gen_launcher(self, sidecar: dict) -> str:
        """Emit the launch_<prefix>() helper for one sidecar."""
        raise NotImplementedError


class CuteDslToolchain(Toolchain):
    """cute.compile + export_to_c: a .o kernel object plus a header of
    per-tensor ABI structs ({data, dynamic shape/stride slots}); module
    (cubin) load is explicit and eager across devices."""

    kind = "cutedsl"
    artifact_exts = (".o", ".h")
    link_source_globs = ("*/*.o",)
    launcher_includes = ()  # per-kernel header, included by prefix below

    REQUIRED_BUILD_KEYS = ("fn", "fake_args", "tensor_args")

    LAUNCHER_TMPL = """\
{prefix}_Kernel_Module_t {prefix}_module;
c10::once_flag {prefix}_loaded;

void launch_{prefix}({tparams}, cudaStream_t stream) {{
  c10::call_once({prefix}_loaded, [] {{ {prefix}_Kernel_Module_Load(&{prefix}_module); }});
{fills}
  int32_t rc = cute_dsl_{prefix}_wrapper(&{prefix}_module, {call_args}, stream);
  TORCH_CHECK(rc == 0, "{prefix} launch failed with code ", rc);
}}
"""

    def export(self, b: dict, out_dir: str, arch: str | None = None) -> dict:
        # arch: handled process-wide via CUTE_DSL_ARCH (set by export.py
        # before any cutlass import; the DSL caches it at first read).
        import cutlass.cute as cute

        # Optional compile options (e.g. --enable-assertions). Must match
        # what the op's JIT wrapper passes (minus --enable-tvm-ffi, which
        # would change the exported ABI) or the two routes' SASS diverges.
        opts = b.get("options")
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
                fills.append(f"  {n}_s.data = {n}.data_ptr();")
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
