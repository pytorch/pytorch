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

import glob
import os
import re


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


class TritonToolchain(Toolchain):
    """triton.tools.compile: a self-contained .c (cubin embedded, grid
    baked in) with a flat CUresult entry point; cubin load is lazy inside
    the entry and current-context-only (multi-GPU unsupported for now)."""

    kind = "triton"
    artifact_exts = (".c", ".h")
    link_source_globs = ("*/*.c",)
    launcher_includes = ("#include <cuda.h>",)

    REQUIRED_BUILD_KEYS = ("kernel_path", "kernel_name", "signature", "grid", "args")

    # Entry points are C symbols (the specialization-hash suffix is
    # recorded in the sidecar as "symbol" at export time).
    LAUNCHER_TMPL = """\
extern "C" CUresult {symbol}(CUstream stream, {csig});

void launch_{prefix}({tparams}, cudaStream_t stream) {{
  CUresult rc = {symbol}(stream, {call_args});
  TORCH_CHECK(rc == CUDA_SUCCESS, "{prefix} launch failed with CUresult ", static_cast<int>(rc));
}}
"""

    def export(self, b: dict, out_dir: str, arch: str | None = None) -> dict:
        from pathlib import Path

        from triton.tools.compile import compile_kernel, CompileArgs

        # Explicit arch: activate a fixed-target driver so neither
        # create_binder nor compile touches a device. (CompileArgs.target
        # as a string is unusable: compile.py str-splits it into a
        # GPUTarget with a STR arch, which breaks ptxas selection.)
        if arch:
            self._activate_triton_target(arch)
        compile_kernel(
            CompileArgs(
                path=b["kernel_path"],
                kernel_name=b["kernel_name"],
                signature=b["signature"],
                grid=b["grid"],
                num_warps=b.get("num_warps", 4),
                num_stages=b.get("num_stages", 3),
                out_name=b["prefix"],
                out_path=Path(out_dir) / b["prefix"],
            )
        )
        # compile_kernel appends a specialization-hash suffix to both the
        # file names and the C symbol; normalize the files to
        # <prefix>.{c,h} and record the real symbol for the launcher.
        prefix = b["prefix"]
        symbol = None
        for path in glob.glob(os.path.join(out_dir, f"{prefix}.*_*.*")):
            ext = os.path.splitext(path)[1]
            dst = os.path.join(out_dir, prefix + ext)
            os.replace(path, dst)
            if ext == ".h":
                with open(dst) as hf:
                    m = re.search(rf"CUresult (\w*{prefix}\w*)\(CUstream", hf.read())
                symbol = m.group(1) if m else None
        if symbol is None:
            raise RuntimeError(
                f"{prefix}: could not find entry symbol in generated header"
            )
        return {"symbol": symbol, "args": b["args"]}

    def kernel_includes(self, sidecar: dict) -> list[str]:
        return []  # cubin load is inside the .c; only <cuda.h> types needed

    def gen_launcher(self, sidecar: dict) -> str:
        prefix = sidecar["prefix"]
        args = sidecar["args"]
        csig, tparams, call = [], [], []
        for a in args:
            if a["kind"] == "tensor":
                csig.append(f"CUdeviceptr {a['name']}")
                tparams.append(f"const at::Tensor& {a['name']}")
                # read_only inputs go through const_data_ptr: a mutable
                # data_ptr() would materialize copy-on-write tensors.
                ptr = (
                    f"{a['name']}.const_data_ptr()"
                    if a.get("read_only")
                    else f"{a['name']}.data_ptr()"
                )
                call.append(f"reinterpret_cast<CUdeviceptr>({ptr})")
            else:
                csig.append(f"{a['ctype']} {a['name']}")
                tparams.append(f"{a['ctype']} {a['name']}")
                call.append(a["name"])
        return self.LAUNCHER_TMPL.format(
            prefix=prefix,
            symbol=sidecar["symbol"],
            csig=", ".join(csig),
            tparams=", ".join(tparams),
            call_args=", ".join(call),
        )


class TritonFromCubinToolchain(TritonToolchain):
    """Triton compiled to a RAW cubin, launched by a generic driver-API
    launcher that this toolchain generates -- no triton.tools.compile C
    template. Spike for the "generic cubin launcher" direction: any
    compiler that yields SASS + (symbol, shared bytes) metadata can share
    this launcher shape. Fixes the C-template path's two limitations:
    module load is per-device (call_once over cuModuleLoadData at first
    use on each device) and errors surface as TORCH_CHECK, not exit().

    Builder contract: same keys as TritonToolchain, minus ``grid``
    (string baked into C) and plus ``launch``: {"grid_x"/"grid_y"/
    "grid_z": C++ exprs over the named scalar args, "block":
    num_warps*32 threads is derived}. The cubin is embedded in the
    generated .cpp as a byte array, so deployment stays a single .so.
    """

    kind = "triton_cubin"
    artifact_exts = (".cubin",)
    link_source_globs = ()  # cubin bytes are embedded in the generated .cpp
    launcher_includes = ("#include <cuda.h>",)

    REQUIRED_BUILD_KEYS = ("kernel_path", "kernel_name", "signature", "launch", "args")

    LAUNCHER_TMPL = """\
namespace {{
// {prefix}: raw cubin ({cubin_len} bytes), embedded; loaded per device on
// first use. Generic driver-API launch: any SASS-emitting toolchain can
// share this shape.
const unsigned char {prefix}_cubin[] = {{{cubin_bytes}}};
constexpr int kMaxDevices = 64;
CUfunction {prefix}_fn[kMaxDevices] = {{}};
c10::once_flag {prefix}_once[kMaxDevices];

CUfunction {prefix}_get(int device) {{
  TORCH_CHECK(device >= 0 && device < kMaxDevices, "device index ", device);
  c10::call_once({prefix}_once[device], [&] {{
    CUmodule mod = nullptr;
    CUresult rc = cuModuleLoadData(&mod, {prefix}_cubin);
    TORCH_CHECK(rc == CUDA_SUCCESS, "{prefix}: cuModuleLoadData failed with CUresult ", static_cast<int>(rc));
    rc = cuModuleGetFunction(&{prefix}_fn[device], mod, "{symbol}");
    TORCH_CHECK(rc == CUDA_SUCCESS, "{prefix}: cuModuleGetFunction failed with CUresult ", static_cast<int>(rc));
  }});
  return {prefix}_fn[device];
}}
}} // namespace

void launch_{prefix}({tparams}, cudaStream_t stream) {{
{arg_decls}
  // Triton kernel ABI appends two hidden scratch pointers after the
  // visible arguments (see triton/tools/compile.py arg_pointers).
  CUdeviceptr global_scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void* kernel_args[] = {{{arg_ptrs}, &global_scratch, &profile_scratch}};
  const unsigned gx = {grid_x};
  const unsigned gy = {grid_y};
  const unsigned gz = {grid_z};
  if (gx * gy * gz == 0) return;
  int device = -1;
  TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "{prefix}: cudaGetDevice failed");
  CUresult rc = cuLaunchKernel({prefix}_get(device), gx, gy, gz,
                               {block_x}, 1, 1, {shared}, stream, kernel_args, nullptr);
  TORCH_CHECK(rc == CUDA_SUCCESS, "{prefix} launch failed with CUresult ", static_cast<int>(rc));
}}
"""

    def export(self, b: dict, out_dir: str, arch: str | None = None) -> dict:
        import triton
        from triton.backends.compiler import GPUTarget

        kernel_mod = _load_module_by_path("kernel", b["kernel_path"])
        kernel = getattr(kernel_mod, b["kernel_name"])

        sig = [s.strip() for s in b["signature"].split(",")]

        def _const(s: str):
            try:
                return int(s)
            except ValueError:
                return None

        constants = {
            kernel.arg_names[i]: _const(s)
            for i, s in enumerate(sig)
            if _const(s) is not None
        }
        arg_types = {
            kernel.arg_names[i]: s for i, s in enumerate(sig) if _const(s) is None
        }

        if arch:
            target = self._activate_triton_target(arch)
        else:
            import torch

            cap = torch.cuda.get_device_capability()
            target = GPUTarget("cuda", cap[0] * 10 + cap[1], 32)
        src = triton.compiler.ASTSource(
            fn=kernel, constexprs=constants, signature=arg_types
        )
        compiled = triton.compile(
            src,
            target=target,
            options={
                "num_warps": b.get("num_warps", 4),
                "num_stages": b.get("num_stages", 3),
            },
        )
        with open(os.path.join(out_dir, b["prefix"] + ".cubin"), "wb") as f:
            f.write(compiled.asm["cubin"])
        return {
            "args": b["args"],
            "launch": b["launch"],
            "symbol": compiled.metadata.name,
            "shared": compiled.metadata.shared,
            "block_x": 32 * b.get("num_warps", 4),
        }

    def kernel_includes(self, sidecar: dict) -> list[str]:
        return []

    def gen_launcher(self, sidecar: dict) -> str:
        # The cubin is read at GENERATION time from next to the sidecar
        # and embedded; the sidecar carries its own directory.
        cubin_path = os.path.join(sidecar["_dir"], sidecar["prefix"] + ".cubin")
        with open(cubin_path, "rb") as f:
            data = f.read()
        cubin_bytes = ",".join(str(x) for x in data)

        args = sidecar["args"]
        tparams, decls, ptrs = [], [], []
        for a in args:
            n = a["name"]
            if a["kind"] == "tensor":
                tparams.append(f"const at::Tensor& {n}")
                # read_only inputs go through const_data_ptr: a mutable
                # data_ptr() would materialize copy-on-write tensors.
                ptr = (
                    f"{n}.const_data_ptr()" if a.get("read_only") else f"{n}.data_ptr()"
                )
                decls.append(
                    f"  CUdeviceptr {n}_p = reinterpret_cast<CUdeviceptr>({ptr});"
                )
                ptrs.append(f"&{n}_p")
            else:
                tparams.append(f"{a['ctype']} {n}")
                ptrs.append(f"&{n}")
        launch = sidecar["launch"]
        return self.LAUNCHER_TMPL.format(
            prefix=sidecar["prefix"],
            symbol=sidecar["symbol"],
            cubin_len=len(data),
            cubin_bytes=cubin_bytes,
            tparams=", ".join(tparams),
            arg_decls="\n".join(decls),
            arg_ptrs=", ".join(ptrs),
            grid_x=launch["grid_x"],
            grid_y=launch.get("grid_y", "1"),
            grid_z=launch.get("grid_z", "1"),
            block_x=sidecar["block_x"],
            shared=sidecar["shared"],
        )


def _load_module_by_path(name: str, path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOOLCHAINS: dict[str, Toolchain] = {
    tc.kind: tc
    for tc in (CuteDslToolchain(), TritonToolchain(), TritonFromCubinToolchain())
}


def get_toolchain(kind: str) -> Toolchain:
    if kind not in TOOLCHAINS:
        raise RuntimeError(
            f"unknown toolchain kind {kind!r}; known: {sorted(TOOLCHAINS)}"
        )
    return TOOLCHAINS[kind]
