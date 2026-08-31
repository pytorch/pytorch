# Native AOT: embedding compiled DSL kernels into ATen

This document describes the "native-AOT" system: DSL kernels (CuTeDSL, Triton)
compiled ahead of time and embedded into the generated ATen structured-op
wrappers, so that a chosen set of shapes/dtypes is served by a DSL kernel with
no Python on the call path and no runtime compilation.

It is both a design doc (part 1-3) and a how-to (part 4 onwards). If you only
want to add an op, read [The declaration contract](#4-the-declaration-contract)
and [Walkthrough: a new AOT op](#5-walkthrough-a-new-aot-op), then copy
`ops/topk/aot.py`, the declaration this stack ships.

**"Native AOT" is unrelated to AOTInductor / AOTAutograd.** There is no graph
capture here. The unit of work is a single eager aten op, and the mechanism is a
`DispatchStub` consulted from inside that op's generated structured wrapper. The
name is `torch._native` (the DSL override stack) plus "aot".

---

## 1. What this is and why it exists

`torch._native` already lets a DSL kernel override an aten op at runtime: a
Python router evaluates a predicate (`cond`) per call and, when it matches,
dispatches to a JIT-compiled DSL kernel. See `torch/_native/README.md` for that
layer. It wins on kernel quality and on developer experience (fast iteration,
easy import of state-of-the-art kernels), but it loses on three things:

* **Recompiles are unpredictable and expensive.** From the design doc: an ads
  SEV where an `index_add` recompile on a new sequence length cost ~700 ms in
  some iterations, about a 33% end-to-end slowdown.
* **Start-up cost.** Single-kernel compiles are O(100 ms), but a model has many
  of them, and heavily-unrolled kernels (top-k) can be O(seconds).
* **Host overhead.** The JIT route has a constant per-launch floor of roughly
  15-20 us: the conditional ("should I run a DSL kernel"), the recompile check
  ("do I need to build this kernel"), and launch overhead. Even when the DSL
  kernel is faster, small inputs come out slower overall.

Native AOT is the hybrid answer, not a replacement:

* For a chosen grid of "common" cases, AOT-compile the DSL kernel into the aten
  library. No runtime compilation, and the dispatch logic moves into C++.
* Everything else stays on the Python JIT route exactly as before.
* Calls that neither layer claims fall through to stock aten, also as before.

The canonical shape of a coverage decision (from the design doc): "RMSNorm:
instantiate for N = [4096, 6144, 8192] and dtypes = [float32, bfloat16], all
others JIT compiled at runtime."

### Where this is going

Today native AOT is a hybrid: a chosen grid is served AOT, everything else JIT,
everything after that stock aten. The design doc's proposal is stronger: for
future architectures (Rubin is named), DSL-provided ops become the AOT
*source of truth* for whole op families (reductions, pointwise, topk) and those
kernels are not built in regular ATen for those arches at all, with a reach
proposal to cut Hopper+ (or Blackwell+) over to DSL-provided AOT ops.

Nothing in this document assumes that end state. The stub declines to `op.impl`
today, and "there is always a stock kernel underneath" is load-bearing
everywhere below (the fallback invariants in section 2, the arch gate's
fall-back-rather-than-fail rationale in section 7). But the proposal explains
why coverage is deliberately conservative: a wrong decline costs performance,
never correctness.

### Measured effect

All numbers below are quoted from the design doc's measurement tables. They are
end-to-end microseconds per call. Read them as "the AOT layer removes the JIT
route's host penalty", not as a claim that AOT always wins: what the kernel does
once launched is still the declaration's problem, and the e2e table below has
rows where AOT loses to ATen for exactly that reason.

They were measured on the full prototype tree, which declared 41 ops across 6
modules. This stack ships one declaration (see
[Current scope](#current-scope)), so every row other than topk's measures a
prototype declaration that is not in this tree. They are kept because what they
measure is the layer, which is what these commits add.

BMM outer-product (Triton), host cost per route -- queue-independent, flat
across sizes:

| route                     | host us/call | vs ATen |
| ------------------------- | ------------ | ------- |
| ATen (cublas dispatch)    | ~9.1         | --      |
| AOT (embedded DSL kernel) | ~10.5        | +1.4    |
| JIT (Python router)       | ~37.3        | +28.2   |

The same op end to end (device time identical for JIT and AOT on these shapes,
so the spread is host cost plus kernel quality):

| B | ATen | JIT | AOT | AOT vs ATen | AOT vs JIT |
| --- | --- | --- | --- | --- | --- |
| 8 | 9.6 | 35.8 | 10.6 | 0.90x | 3.37x |
| 64 | 9.6 | 34.8 | 10.5 | 0.91x | 3.30x |
| 512 | 14.2 | 33.6 | 11.2 | 1.26x | 3.00x |
| 4096 | 94.6 | 45.0 | 45.0 | 2.10x | 1.00x |
| 32768 | 576.2 | 338.8 | 338.7 | 1.70x | 1.00x |

Note the two rows below 1.0x. The prototype's `bmm_outer_product` declaration
neither shipped a small-problem kernel nor gated on too-small problems, so at
small B the AOT route is slower than ATen. That is a kernel/gating deficiency in
the declaration, not a cost of the layer. **Read it as an instruction: a
declaration must gate the sizes where
its kernel loses.** topk's prelude does exactly that with its full-wave check
(see [The declaration, annotated](#44-the-declaration-annotated)).

CuTeDSL ops on B200, e2e us/call on covered shapes:

```
op (covered shape)              |   aten |    JIT |    AOT | AOT-aten | JIT-AOT
--------------------------------+--------+--------+--------+----------+--------
topk M=1024, N=4096, K=64       |  100.0 |   37.2 |   33.4 |    -66.6 |    +3.8
topk M=256,  N=4096, K=64       |  109.1 |   29.8 |   18.2 |    -90.9 |   +11.6
scatter_add M=2048, N=512       |   13.5 |   61.6 |   17.8 |     +4.3 |   +43.8
pw add 64K fp32                 |    7.7 |   36.5 |   13.9 |     +6.2 |   +22.6
pw sigmoid 64K bf16             |    6.8 |   26.4 |   12.6 |     +5.8 |   +13.8
red sum 512x4096 fp32           |    8.6 |   81.8 |   13.1 |     +4.5 |   +68.7
```

Per-call AOT premium, in-process ablation (e2e us):

```
layer                                    | pw add | scatter_add
-----------------------------------------+--------+------------
stock aten                               |    5.9 |         9.8
C++ stub only (Python router off)        |    5.7 |         9.7
+ Python router entry (covers=True)      |   +2.0 |        +3.1
+ real covers check                      |   +3.8 |        +1.3
                                           (Python)  (cpp_covers)
```

Conclusions the doc draws from this, which matter when you tune an op:

* The C++ AOT machinery is free: stub-only is at or below stock aten on every
  op measured. A CUPTI profile shows exactly one kernel launch per call, with
  no per-call module lookups, occupancy queries or descriptor re-encodes; the
  generated launcher body is sub-0.2 us (one `c10::call_once` atomic load, ABI
  struct fills from `size()`/`stride()`, an rc check). Module/cubin load happens
  once per process behind the once-flag.
* The entire remaining +4-6 us premium over stock aten is the Python
  JIT-router round trip: ~2-3 us router entry plus the coverage check. Writing
  `cpp_covers` cuts the coverage share from ~3.8 us to ~1.3 us.
* topk on covered shapes is 3-6x faster end-to-end than aten. The
  scatter_add / pointwise / reduction rows are aten-host-bound at those sizes:
  the premium is visible, the AOT kernels themselves match.

Methodology for the CuTeDSL table, verbatim from the doc: "fresh stage-2 rebuild
from HEAD 66b3157268ac, all JIT disk caches purged, subprocess per (op, route),
each child asserts its route via profiler kernel names before measuring." The
host columns are median profiler `cpu_time` of the `aten::<op>` event and
over-read by about 0.5 us per nested profiler event; aten topk's host number is
regime-dependent (72-121 us) because it enqueues ~10 kernels per call and the
launch-queue stall is charged to `cpu_time`.

### Current scope

One declaration, `ops/topk/aot.py` (`CUDA`), expanding to 48 precompiled kernels
per arch: fp32 and bf16 radix top-k over N in {2048, 4096, 8192, 16384}, K in
{64, 128, 256}, in both determinism modes. It exports `cpp_covers`; nothing here
exports `cpp_helpers`.

A build compiles that grid once per eligible arch, so what ships depends on
`TORCH_CUDA_ARCH_LIST`: a manywheel CUDA 13.x list (`7.5;8.0;8.6;9.0;10.0;12.0`)
admits `sm_90` and `sm_100` and carries 96 kernels, `b200-native-aot.yml`
(`10.0a`) carries 48, and CUDA 12.x builds skip stage 2 entirely.

---

## 2. Architecture

### 2.1 The mechanism: a DispatchStub between meta() and impl()

torchgen generates, per declaration, one `at::native` `DispatchStub` whose
signature *is* the op's structured impl signature and whose return type is
`bool`, with **no kernel registered by default**:

```cpp
// build/aten/src/ATen/NativeAotStubs.h (generated from
// aten/src/ATen/templates/NativeAotStubs.h)
using topk_aot_fn = bool (*)(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, const at::Tensor & values, const at::Tensor & indices);
DECLARE_DISPATCH(topk_aot_fn, topk_aot_stub)
```

```cpp
// build/aten/src/ATen/NativeAotStubs.cpp
DEFINE_DISPATCH(topk_aot_stub);
REGISTER_NO_CPU_DISPATCH(topk_aot_stub)
```

It then splices a single consultation line into the generated structured
wrapper, between `op.meta(...)` and `op.impl(...)`:

```cpp
// build/aten/src/ATen/RegisterCUDA_0.cpp (generated; the functional
// variant of aten::acos)
at::Tensor wrapper_CUDA_acos(const at::Tensor & self) {
  // No device check
structured_acos_out_functional op;
op.meta(self);
if (!(at::globalContext().allowNativeAot() && at::native::acos_aot_stub.is_device_supported(c10::DeviceType::CUDA) && at::native::acos_aot_stub(c10::DeviceType::CUDA, self, op.outputs_[0]))) { op.impl(self, op.outputs_[0]); }
return std::move(op.outputs_[0]);
}
```

The emitter is short enough to read whole:

```python
# torchgen/native_aot.py
def gen_stub_consultation(m: NativeAotManifest, impl_exprs: str) -> str:
    """The structured-wrapper call site. The stub has no kernel unless the
    AOT library registered one, and the Context switch gates the whole
    path; a true return means the AOT kernel filled the meta()-allocated
    outputs and op.impl is skipped."""
    device_type = f"c10::DeviceType::{m.dispatch_key}"
    stub = f"at::native::{m.stub_name()}"
    return (
        f"if (!(at::globalContext().allowNativeAot() && "
        f"{stub}.is_device_supported({device_type}) && "
        f"{stub}({device_type}, {impl_exprs}))) {{ op.impl({impl_exprs}); }}"
    )
```

Three consequences to internalize:

1. **Only structured ops can be embedded.** The consultation has to sit between
   `meta()` and `impl()`; an unstructured op has nowhere to put it.
2. **Outputs already exist when your kernel runs.** `meta()` allocated and
   validated them and the device guard is held, so an AOT kernel body only
   fills outputs and returns `true`. Returning `false` runs `op.impl` -- the
   stock kernel -- unchanged.
3. **A stock wheel is behaviorally identical.** With no AOT library linked, the
   stub pointer is null, `is_device_supported` is false, and the cost is a
   relaxed atomic load plus a null check. The commit that landed the codegen
   measured this as indistinguishable from noise (<0.1 us).

All three schema kinds of a structured group funnel through the same stub; only
the output expression differs (`op.outputs_[0]` for functional and inplace,
`op.maybe_get_output(0)` for `out=`).

### 2.2 The dispatch path at runtime

Two gates. The first subtracts AOT coverage from the Python JIT layer; the
second is the stub consultation above.

```
Op call -> dispatcher -> Python eager_router (exists only if the op has a JIT override)
  |
  |  GATE 1 (router head, once per call): coverage.covers(args, kwargs)
  |    prefers torch.ops._native_aot.covers_<decl_id>   <- your cpp_covers, compiled
  |    else covered_axes(*args) + grid match            <- your Python
  |
  |-- covered -> decline ALL JIT conds -> aten CUDA wrapper
  |     wrapper: meta() allocates outputs
  |     GATE 2: stub consultation -> <op>_cuda_aot_kernel
  |         arch gate -> your prelude -> first matching dispatch branch -> your launch
  |         -> returns true, op.impl skipped                    = THE AOT HIT
  |       (prelude or all branches reject -> return false -> op.impl = stock aten)
  |
  `-- not covered -> JIT conds evaluated normally
        a cond matches -> JIT impl (compiled at runtime, cached)  = JIT path
        none match     -> aten wrapper -> stub declines -> stock aten
```

Gate 1 lives in the router:

```python
# torch/_native/registry.py
    # Calls covered by AOT kernels embedded in the aten implementation
    # (see torch/_native/ops/<op>/aot.py) must decline the JIT route:
    # the router's no-match fallback lands in the aten kernel, whose
    # native-AOT stub serves them. Checked once per call here rather
    # than per cond -- every override of the op would get the same
    # answer for the same arguments, and ops with several paths (e.g.
    # scatter_add's TMA + vec-scatter) would pay covers() N times.
    # Applies to unconditional overrides too. None when the op has no
    # AOT declaration, so uncovered ops pay nothing.
    from . import aot_manifest

    coverage = aot_manifest.get_coverage(op_symbol, dispatch_key)

    def _dispatch(args, kwargs, swallow_cond_exceptions: bool):
        # covers() degrades covered_axes exceptions to "uncovered", so
        # this is safe under FakeTensor in the compile router too.
        if coverage is not None and coverage.covers(args, kwargs):
            return _NO_MATCH
```

and the coverage rule itself is:

```python
# torch/_native/aot_manifest.py
    def covers(self, args: tuple, kwargs: dict) -> bool:
        # Gate on the Context switch the stub consultations check: with
        # AOT masked (set_aot_enabled(False)), a covered call must keep
        # its JIT route rather than decline into a stub that will not
        # fire -- otherwise masking silently loses BOTH accelerated
        # routes. ~0.1us per call.
        if not torch._C._get_native_aot_enabled():
            return False
        cpp = self._resolve_cpp_covers()
        if cpp is not None:
            try:
                return cpp(*args, **kwargs)
            except Exception:
                # Arguments the schema can't bind (SymInt sizes, exotic
                # kwargs): uncovered; the JIT cond decides.
                return False
        try:
            values = self._covered_axes(*args, **kwargs)
        except Exception:
            return False
        for point in self._grid:
            if all(
                self._field_matches(values.get(f), v)
                for f, v in point.items()
                if f in values
            ):
                return True
        return False
```

So: **a call is covered iff some expanded grid point agrees with every field
`covered_axes()` returns.** Grid fields your `covered_axes` does not return
(block sizes, M buckets) are ignored. Dtypes are compared canonically (grid
strings like `"float32"` against `torch.float32`). Any exception on either path
degrades to "uncovered", which is what makes the router safe under FakeTensor.

The two sides need not agree exactly. The C++ dispatch chain is the authority on
what actually launches, and drift is benign: a call both sides decline lands on
stock aten. But coverage must never be **wider** than the stub's acceptance,
because a covered call has already given up its JIT route.

### 2.3 The two-stage build

Stage 2 exists because kernel builder modules import torch while torchgen runs
before torch exists. That is a consequence of a design choice, not a law: the
original design required kernels to be pure DSL with no torch dependency, which
permits a **one-stage** build (torch plus AOT ops in one pass) and, per the
design doc, "feels cleanest". The pointwise and reduction kernel modules made
that restriction untenable -- their kernels are entangled with torch-side
plumbing -- so it was relaxed and the build was split. (Those are JIT overrides
under `torch/_native/ops/`; neither carries an AOT declaration in this stack.)
Going back is possible and would be a large amount of op-side work; see
[Known future work](#known-future-work).

**Stage 1 -- the normal torch build.** torchgen walks every
`torch/_native/ops/*/aot.py`, validates it, and emits `NativeAotStubs.{h,cpp}`
plus the consultation lines. Nothing is compiled: the stubs are null and the
generated AOT kernel functions do not exist yet. A torch built and stopped here
is fully functional and exercises zero AOT kernels. Note where the two halves
land: `NativeAotStubs.cpp` compiles into **torch_cpu** (that is where
`<op>_aot_stub` is defined) while the consultation lines live in the CUDA
register files in **torch_cuda**, which is why adding a declaration needs a
stage-1 build and not just stage 2's torch_cuda relink. Declarations are wired in
as codegen inputs, so editing one re-runs torchgen:

```cmake
# cmake/Codegen.cmake
  # Native-AOT declarations are codegen inputs: they add DispatchStub
  # declarations and structured-wrapper call sites (torchgen/native_aot.py,
  # contract in tools/native_aot/decl.py -- also a codegen input).
  file(GLOB native_aot_manifests CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_LIST_DIR}/../torch/_native/ops/*/aot.py"
       "${CMAKE_CURRENT_LIST_DIR}/../tools/native_aot/decl.py")
```

**Stage 2 -- export, generate, relink.** Runs against the built and *installed*
torch (`tools/native_aot/build_stage2.py`):

1. `export.py` expands each declaration's grid and, per point, calls the
   builder's `build(spec)` and the toolchain's compile/export, writing
   `<prefix>.{h,o}` (CuTeDSL) or `<prefix>.{c,h}` / `<prefix>.cubin` (Triton)
   plus a `<prefix>.json` sidecar into `build/native_aot/<decl_id>/`.
2. `gen_aot_lib.py` turns declarations + sidecars into one
   `aot_<decl_id>_<key>.cpp` per declaration: the launchers, the stub kernel
   (arch gate, prelude, dispatch chain), the `REGISTER_<KEY>_DISPATCH` that
   fills stage 1's null stub, and the `covers_<decl_id>` custom op.
3. `cmake --build . --target torch_cuda` relinks with those sources globbed in,
   and the relinked `libtorch_cuda.so` is copied over the installed one (and
   optionally patched into a built wheel).

The sidecar is the channel from export-time knowledge to codegen: it records the
exact spec each artifact was compiled for, which is what lets `gen_aot_lib` emit
one dispatch branch per *shipped* kernel and refuse stale pairings.

Note a deliberate divergence from the design doc, which specified the AOT/JIT
handover as sidecar-driven shared state. It is not: sidecars never reach
runtime. Coverage is computed from the declaration's grid alone
(`aot_manifest.py` reads no sidecar; it builds `_Coverage(d.ATEN_OP,
d.covered_axes, expand_specs(d.kernel_precompile_grid()))`), with only
`expand_specs` shared between export and runtime. The consequence is recorded
under [Limitations](#10-limitations-and-restrictions): a grid point the export
skipped is still "covered", so such a call declines JIT and lands on stock aten.
Making coverage sidecar-aware is the fix if that ever bites; it would cost a
per-process artifact scan.

```python
# torchgen/native_aot_spec_grid.py
"""Spec-grid expansion shared by the AOT export tool and runtime coverage.

Lives in torchgen because both consumers need it when torch may not be
importable, and torchgen is pure Python, ships in the wheel, and is
already a torch import-time dependency. tools/native_aot/export.py and
torch._native.aot_manifest both import it normally.
"""

import itertools


def expand_specs(specs: list[dict]) -> list[dict]:
    """Cross-multiply list-valued fields of each spec block; concatenate
    blocks. Scalars are singleton axes."""
    points = []
    for spec in specs:
        keys = list(spec.keys())
        axes = [v if isinstance(v, list) else [v] for v in spec.values()]
        points.extend(dict(zip(keys, combo)) for combo in itertools.product(*axes))
    return points
```

Build-time budget, from the design doc, on a 1-GPU dev machine: kernel compile
9.2 s wall (176 kernels, 36 cores); launcher and stub generation 10.2 s; build
41 TUs and relink the .so 8 s; copy the relinked `libtorch_cuda.so` 1 s. Total
28.5 s wallclock. Those counts are the prototype's whole tree; topk alone is 48
kernels in one TU per arch.

### 2.4 Toolchains

The DSL-specific parts are confined to one class per DSL:

```python
# tools/native_aot/toolchains.py
  3. ``gen_launcher(sidecar)`` emits C++ with the toolchain-independent
     signature every manifest ``body`` programs against:

         void launch_<prefix>(const at::Tensor&..., <scalars>..., c10::Stream)

     Everything above the launcher (guard chain, cond, DispatchStub
     registration) is toolchain-blind.
```

A DSL is embeddable if it can produce one of three artifact shapes (the design
doc's taxonomy, for CUDA): (1) a **.o** with a known call signature we link
directly; (2) **source** (`.cpp`/`.cu` plus a header) we compile into the
library; (3) a **cubin** we embed and launch ourselves. The three registered
kinds are one of each, which is why the abstraction is `artifact_exts` plus
`link_source_globs` plus `gen_launcher`. Of the DSLs the doc considered: CuTeDSL
takes path 1 (or 3), Triton path 2, cuTile path 3, and Helion emits another DSL,
so it inherits whichever path its backend takes.

Registered kinds: `cutedsl` (`cute.compile` + `export_to_c` -> `.o` plus an ABI
header, module load explicit and eager across devices) and `triton` (compiled to
a raw cubin, embedded as bytes in the generated `.cpp`, with a launcher this
toolchain writes: per-device module load and `TORCH_CHECK` errors).
`triton.tools.compile`'s C template is deliberately **not** used -- it calls
`cuModuleLoadData`/`cuLaunchKernel` directly, and being triton-generated it
cannot be routed through `c10::cuda::DriverAPI`, so linking it would force a
`libcuda` dependency on `torch_cuda` and break CUDA builds on driverless
machines. Which kind a point uses
is chosen by the **builder's returned dict** (`kind`, defaulting to
`"cutedsl"`), not by the declaration. Adding a DSL is one class plus a
registry entry; `export.py`, `gen_aot_lib.py` and the CMake project are
untouched.

### 2.5 CMake integration

CMake contributes one line, and makes no decisions:

```cmake
# caffe2/CMakeLists.txt
  include("${CMAKE_BINARY_DIR}/native_aot/native_aot.cmake" OPTIONAL)
```

Everything else is emitted by `gen_aot_lib.write_cmake_include()`, which knows
what it just generated and so needs no globbing, no manifest to parse and no
staleness comparison: the file lists exact paths, and stage 2 deletes it before
writing new sources, so a stale one cannot survive. `OPTIONAL` is what makes a
tree that never ran stage 2 (or opted out) build normally -- no sources, no
kernels registered, stock aten -- so the main build never depends on stage 2.

The emitted file carries the `TORCH_NATIVE_AOT` opt-out (checked at configure
time, because a previous run's file can still be on disk), registers itself in
`CMAKE_CONFIGURE_DEPENDS`, marks the CuTeDSL objects `EXTERNAL_OBJECT` so CMake
links rather than compiles them, adds each artifact subdirectory as an include
dir (the generated `.cpp` includes the exported ABI headers by bare name), sets
`BUILD_WITH_INSTALL_RPATH`, and links `libcuda_dialect_runtime_static.a` with
`--exclude-libs` plus the generated version script. Because it lists paths rather
than globbing, multi-arch export needs no CMake change: nesting depth is the
generator's problem (see [Arch gating](#7-arch-gating)).

---

## 3. What gets generated, end to end

For `aten::topk @ CUDA`, stage 2 writes `build/native_aot/topk/aot_topk_cuda.cpp`
containing, in order:

```cpp
// build/native_aot/topk/aot_topk_cuda.cpp (generated; abridged)
// @generated by tools/native_aot/gen_aot_lib.py from
// torch/_native/ops/topk/aot.py -- do not edit
//
// Structured META precomputes NOTHING for this op: schema args (incl. any dim) arrive RAW -- wrap dims before comparing.
#include <ATen/ATen.h>
#include <ATen/NativeAotStubs.h>
// ... plus TensorIterator.h, CUDAContext.h, torch/library.h, <algorithm>, <limits>
#include "topk_radix_bf16_n2048_k128_det.h"    // one per exported kernel

namespace {

// one launcher per exported kernel, emitted by the toolchain
topk_radix_bf16_n2048_k128_det_Kernel_Module_t topk_radix_bf16_n2048_k128_det_module;
c10::once_flag topk_radix_bf16_n2048_k128_det_loaded;

void launch_topk_radix_bf16_n2048_k128_det(const at::Tensor& mX, const at::Tensor& mValues, const at::Tensor& mIndices, c10::Stream stream) {
  c10::call_once(topk_radix_bf16_n2048_k128_det_loaded, [] { topk_radix_bf16_n2048_k128_det_Kernel_Module_Load(&topk_radix_bf16_n2048_k128_det_module); });
  topk_radix_bf16_n2048_k128_det_Tensor_mX_t mX_s;
  mX_s.data = const_cast<void*>(mX.const_data_ptr());
  mX_s.dynamic_shapes[0] = static_cast<int32_t>(mX.size(0));
  mX_s.dynamic_strides[0] = mX.stride(0);
  mValues_s.data = mValues.mutable_data_ptr();   // writable outputs
  // ... remaining mValues_s / mIndices_s slots ...
  int32_t rc = cute_dsl_topk_radix_bf16_n2048_k128_det_wrapper(&topk_radix_bf16_n2048_k128_det_module, &mX_s, &mValues_s, &mIndices_s,
                                         c10::cuda::CUDAStream(stream).stream());
  TORCH_CHECK(rc == 0, "topk_radix_bf16_n2048_k128_det launch failed with code ", rc);
}

// the stub kernel: signature == the structured impl signature
bool topk_cuda_aot_kernel(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, const at::Tensor & values, const at::Tensor & indices) {
  // Device gate: declaration ARCHS x shipped artifacts = sm_100a
  if (!(at::cuda::getCurrentDeviceProperties()->major == 10)) return false;
  // Size gate: the DSL's exported ABI carries int32_t shape slots
  // (see _int32_size_gate); a bigger dim would truncate silently.
  if (C10_UNLIKELY(self.sizes().end() != std::find_if(self.sizes().begin(), self.sizes().end(), _naot_dim_too_big))) return false;

        if (self.scalar_type() != at::kFloat && self.scalar_type() != at::kBFloat16) return false;
        // ... the rest of cpp_dispatch_prelude(), verbatim ...
        const int64_t M = self.numel() / N;
        if (M < at::cuda::getCurrentDeviceProperties()->multiProcessorCount) return false;
  if (self.scalar_type() == at::kBFloat16 && N == 16384 && k == 128 && det) {

          auto self_2d = self.view({M, N});
          auto values_2d = values.view({M, k});
          auto indices_2d = indices.view({M, k});
          launch_topk_radix_bf16_n16384_k128_det(self_2d, values_2d, indices_2d, at::cuda::getCurrentCUDAStream());
    return true;
  }
  // ... 47 more branches ...
  return false;
}

bool topk_cuda_covers(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, const std::optional<at::Tensor>& values, const std::optional<at::Tensor>& indices) {
  // Device gate: ... (same gate, injected here too)
  // ... cpp_covers() verbatim ...
}

} // namespace

// Register on the generated DispatchStub (same mechanism as aten's own
// runtime-registered CUDA kernels; see ATen/native/DispatchStub.h).
namespace at::native {
REGISTER_CUDA_DISPATCH(topk_aot_stub, &::topk_cuda_aot_kernel)
} // namespace at::native

TORCH_LIBRARY_FRAGMENT(_native_aot, m) {
  m.def("covers_topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, Tensor? values=None, Tensor? indices=None) -> bool", &::topk_cuda_covers);
}
```

Note the header comment: `gen_aot_lib` states, per op, whether the structured
META precomputes any schema arguments. That is not cosmetic -- see
[Precomputed vs raw dims](#61-precomputed-vs-raw-dims).

---

## 4. The declaration contract

An op opts in by shipping `torch/_native/ops/<op>/aot.py`. The authoritative
contract and the validating loader are `tools/native_aot/decl.py`; read its
module docstring if this section and the code ever disagree.

**Module scope must import with stdlib alone.** torchgen loads the file by path,
before torch exists and without package context. Import torch lazily inside
function bodies. The kernel builder module (`KERNEL_MODULE`) has no such
restriction: it is package-imported in stage 2 with the built torch available,
which is why the AOT and JIT routes can share one kernel body by construction.

A module declares either **one op** (module-level exports) or a **family**
(exports `declarations() -> list` of objects carrying the same exports as
attributes and methods).

### 4.0 Why hooks that return C++ as strings

The design doc considered and rejected two alternatives. A declarative
*expression language* for conditions and launches would be typed and checkable,
but has to grow to cover every predicate a real op needs (TensorIterator
classification, device-property queries, dim wrapping) -- appealing in
principle, unpleasant in practice. A **YAML** schema was the original form and
was abandoned: Python is more pleasant to work with, and it lets one shared
table (`_DTYPES`, `_NS`) be interpolated into both the Python and C++ sides of a
declaration, so the two cannot drift by transcription. The price is paid in this
section's sharp edges: doubled braces in f-strings, cardinality expressed as
arity, and no type checking of the emitted C++ until it compiles.

### 4.1 Required exports

| export | shape | meaning |
| ------ | ----- | ------- |
| `ATEN_OP` | `str` | a **structured** op: base name (`"topk"`) when the base resolves to exactly one structured group, or overload-qualified (`"sum.dim_IntList"`, `"gt.Tensor"`) when overloads have separate structured groups. Dots become underscores in `decl_id`, which names the stub, the generated kernel, the artifact dir and the covers op. |
| `DISPATCH_KEY` | `str` | e.g. `"CUDA"`. |
| `KERNEL_MODULE` | `str` | sibling module **filename, including `.py`** (e.g. `"aot_kernel.py"`), exporting `build(spec)`. Export package-imports `torch._native.ops.<dir>.<stem>` (`export.py`, `kernel_module.removesuffix(".py")`), so the op directory needs an `__init__.py`. A missing `.py` is not caught at load time; it surfaces as an ImportError during export. |
| `kernel_precompile_grid()` | `-> list[dict]` | the artifact grid: list-valued fields cross-multiply, one precompiled kernel per expanded point. Values must survive a JSON round trip. |
| `covered_axes(*schema_args)` | `-> dict` | project a live call onto grid axes. Arguments are the live call's args/kwargs for **any** overload of the structured group (base, `.out`, in-place -- all resolve to this one declaration via `aot_manifest._base_name`), so absorb the out-variant outputs as trailing optionals (`out=None`, or `*args, **kwargs`) or the call raises `TypeError`, which `covers()` silently degrades to "uncovered". Runs once per call in the router; keep it cheap. |
| `cpp_dispatch(spec)` | `-> str` | one boolean C++ expression per point: given a call that passed the prelude, is it served by THIS point? First match wins. |
| `cpp_launch(spec, launch_fn)` | `-> str` | C++ invoking this point's kernel through `launch_fn(...)`. No allocation of outputs, no fallback logic. |

### 4.2 Optional exports

| export | shape | meaning |
| ------ | ----- | ------- |
| `ARCHS` | `tuple[str, ...]` | sm strings the kernels are valid on. Defaults to `("sm_90", "sm_90a", "sm_100", "sm_100a", "sm_103", "sm_103a")`. Export skips arches outside it; codegen emits the runtime device gate. |
| `cpp_dispatch_prelude()` | `-> str \| None` | shared front half of the chain: cheap universal rejects and setup (locals, classifier calls) every branch reads. May also `return true` for degenerate calls the op serves without a kernel. |
| `cpp_helpers()` | `-> str \| None` | C++ shared beyond one op (family classifiers), emitted once per generated file. |
| `cpp_covers()` | `-> str \| None` | fast C++ port of `covered_axes` + grid matching, over the **functional** schema arguments plus the out-variant outputs as trailing optionals. Registered as `torch.ops._native_aot.covers_<decl_id>`. |

### 4.3 Cardinality and the generated shape

Quoting the contract verbatim:

```
Emission cardinality: cpp_helpers once per file, cpp_dispatch_prelude
once per op, cpp_dispatch/cpp_launch once per precompile point. The
generated stub is::

    helpers | prelude -> [if (dispatch) { launch; return true; }]* -> return false

in the op's structured impl scope (outputs allocated by meta(), device
guard held). Dispatch conditions are evaluated ASSUMING the prelude
passed; locals the prelude declares are in scope for dispatch and
launch.
```

Cardinality is enforced by **positional arity**, not by name: no-arg exports are
per op/file, one-argument exports (`spec`) are per point, and `cpp_launch` takes
two (`spec`, `launch_fn`). A mismatch is a load-time `RuntimeError` naming the
file and the export, e.g.

```
.../aot.py: cpp_launch must be per-point (spec-taking), expected 2 positional parameter(s), got 1
```

Note the emission is really once per **exported sidecar**: points that export
skipped (arch filtering, `--ops`) simply produce no branch, so the chain can be a
strict subset of the grid.

### 4.4 The declaration, annotated

This is the teaching example: an annotated variant of
`torch/_native/ops/topk/aot.py` with every export present and its contract
stated inline. It is simplified relative to HEAD (24-point grid, one determinism
mode, no full-wave gate in `cpp_covers`); see the shipped file for current
behavior, and the diff notes underneath.

```python
# an annotated walkthrough of torch/_native/ops/topk/aot.py
#
# Module scope must import with stdlib alone (torchgen loads this
# pre-build by file path); torch is imported lazily inside bodies.
# Contract + validating loader: tools/native_aot/decl.py
#
# A module declares ONE op (as here) or a FAMILY by exporting
# declarations() -> list of objects with these same exports
# (a table-driven family builds its list from one row table).

# ---------------------------------------------------------------- constants

ATEN_OP = "topk"          # STRUCTURED op: base name, or overload-qualified
                          # ("gt.Tensor") when overloads have separate
                          # structured groups (ambiguous bases are a
                          # codegen error)
DISPATCH_KEY = "CUDA"
KERNEL_MODULE = "cutedsl_kernels.py"  # sibling exporting build(spec);
                          # package-imported by the export tool with the
                          # BUILT torch available (two-stage build), so it
                          # lives beside the JIT wrappers and both routes
                          # compile the same kernel class

# Shared tables: define once, interpolate everywhere (Python + C++ sides)
_DTYPES = {"float32": "at::kFloat", "bfloat16": "at::kBFloat16"}
_NS = [2048, 4096, 8192, 16384]
_KS = [64, 128, 256]

# ------------------------------------------------------------- pure Python

def kernel_precompile_grid() -> list[dict]:
    """The artifact grid: one precompiled kernel per expanded point
    (list-valued fields cross-multiply). Consumed by export (fan-out),
    codegen (one dispatch/launch branch per point), and coverage
    (matched against covered_axes)."""
    return [{"dtype": list(_DTYPES), "N": _NS, "K": _KS, "deterministic": False}]


def covered_axes(self, k, dim=-1, largest=True, sorted=True) -> dict:
    """Project a live call onto grid axes; covered (declines the JIT
    route) iff some precompile point matches every returned field.
    Exceptions => uncovered. The router runs this ONCE per call ahead
    of the JIT cond chain: keep cheap (no TensorIterator builds, no
    data_ptr() -- it materializes copy-on-write inputs).

    Arguments are the LIVE CALL's args/kwargs, for any overload of the
    structured group: base, .out and in-place symbols all resolve to
    this one declaration (aot_manifest._base_name). topk registers JIT
    overrides on both "topk" and "topk.values", so this can be reached
    with values=/indices= kwargs -- which this signature does NOT
    accept. It is survivable here only because cpp_covers (below) takes
    the out-variant outputs as trailing optionals and the router
    prefers it on any embedded build; a declaration WITHOUT cpp_covers
    must absorb them itself (out=None, or *args/**kwargs), or every
    out= call raises TypeError and covers() silently degrades it to
    "uncovered"."""
    import torch
    return {
        "dtype": self.dtype,
        "N": self.shape[-1] if self.dim() >= 1 else 0,
        "K": k,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }

# --------------------------------------------------------- C++-generating
# Executed once at library-generation time; each returns C++ source.
# Scope of the emitted code: the op's structured impl signature, after
# meta() (outputs allocated, device guard held). NB: whether dim args
# arrive wrapped or RAW varies per structured group; the generated file
# header states it (topk's dim arrives raw -> maybe_wrap_dim below).
# Cardinality convention (loader-enforced): spec-taking = per precompile
# point, no-arg = per op. Generated shape:
#   [helpers] | [prelude] -> [if (dispatch) { launch; return true; }]* -> return false

def cpp_covers() -> str | None:                 # OPTIONAL
    """Fast C++ port of the covered_axes + grid match, over the op's
    FUNCTIONAL schema args (outputs don't exist at router time).
    Compiled into the AOT library and registered as
    torch.ops._native_aot.covers_topk; the router prefers it over the
    Python matching."""
    dtype_accept = " || ".join(f"st == {t}" for t in _DTYPES.values())
    n_accept = " || ".join(f"N == {n}" for n in _NS)
    k_accept = " || ".join(f"k == {kk}" for kk in _KS)
    return f"""
      const auto st = self.scalar_type();
      if (!({dtype_accept})) return false;
      if (at::globalContext().deterministicAlgorithms()) return false;
      const int64_t N = self.dim() >= 1 ? self.size(-1) : 0;
      return ({n_accept}) && ({k_accept});
    """


def cpp_dispatch_prelude() -> str | None:       # OPTIONAL
    """Shared front half of the chain: cheap universal rejects + setup
    (locals like N/M) every branch reads. Absent => each cpp_dispatch
    must be self-contained. Order zero-checks BEFORE divisions: an
    integer div-by-zero here is a process-killing SIGFPE, not an
    exception."""
    dtype_reject = " && ".join(f"self.scalar_type() != {t}" for t in _DTYPES.values())
    return f"""
      if ({dtype_reject}) return false;
      if (!largest || !sorted) return false;
      if (self.dim() < 1) return false;
      if (c10::maybe_wrap_dim(dim, self.dim()) != self.dim() - 1) return false;
      if (!self.is_contiguous() || !values.is_contiguous() || !indices.is_contiguous()) return false;
      if (at::globalContext().deterministicAlgorithms()) return false;
      const int64_t N = self.size(-1);
      if (N == 0) return false;
      const int64_t M = self.numel() / N;
      // Perf gate: one CTA per row; below a full wave aten wins.
      if (M < at::cuda::getCurrentDeviceProperties()->multiProcessorCount) return false;
    """


def cpp_dispatch(spec) -> str:                  # REQUIRED, per point
    """Given a call that passed the prelude, is it served by THIS point?
    First match wins; any expression (equality, ranges) is fine."""
    return f"self.scalar_type() == {_DTYPES[spec['dtype']]} && N == {spec['N']} && k == {spec['K']}"


def cpp_launch(spec, launch_fn: str) -> str:    # REQUIRED, per point
    """Invoke this point's kernel via launch_fn(...). No allocation, no
    fallback -- the chain's return false IS the fallback."""
    return f"""
      auto self_2d = self.view({{M, N}});
      auto values_2d = values.view({{M, k}});
      auto indices_2d = indices.view({{M, k}});
      {launch_fn}(self_2d, values_2d, indices_2d, at::cuda::getCurrentCUDAStream());
    """


def cpp_helpers() -> str | None:                # OPTIONAL, per file
    """C++ shared beyond one op (e.g. a family's TI-based classifier),
    emitted once per generated file."""
    # Not used for topk
    return None
```

What the shipped `torch/_native/ops/topk/aot.py` does differently:

* `"deterministic": [False, True]` on the grid (48 points): there is a
  bit-exact deterministic specialization, so both modes are AOT'd and
  `cpp_dispatch` keys each point on a `det` local the prelude declares.
* `covered_axes` and `cpp_covers` both mirror the prelude's full-wave gate
  (`M >= multiProcessorCount`), so coverage is never wider than the stub.
* `cpp_covers` rejects non-CUDA tensors explicitly, because the device-property
  query would throw.

### 4.5 The validating loader

`tools/native_aot/decl.py` is the only place that reads declarations
structurally. It checks that the three constants are present strings, that the
four required functions are callable with the right arity, that optional
functions (when not `None`) have the right arity, that `ARCHS` normalizes to a
non-empty tuple of `sm_\d+a?` strings, and that `kernel_precompile_grid()`
returns a non-empty list of dicts. It runs the grid function at load time, so
grid construction must work in a torch-free environment.

`ARCHS` is normalized by **mutating the declaration** (`d.ARCHS = tuple(archs)`),
so declaration objects must be plain mutable instances. Frozen dataclasses and
`__slots__` classes fail here.

Semantic validation of `ATEN_OP` lives in torchgen, and its error messages are
the documentation:

```python
# torchgen/native_aot.py
        elif not names:
            raise RuntimeError(
                f"native-aot declaration for {op}@{key}: {op} is not a "
                f"structured op in native_functions.yaml. The stub is "
                f"consulted between meta() and impl(), so unstructured ops "
                f"are served by the JIT layer only; to embed, structure the "
                f"op upstream first (preferred; e.g. var.correction is a "
                f"candidate)"
            )
        elif len(names) > 1:
            raise RuntimeError(
                f"native-aot declaration for {op}@{key}: base name {op!r} is "
                f"ambiguous across structured groups {names}; qualify "
                f"ATEN_OP with the overload (e.g. {names[0]!r})"
            )
```

---

## 5. Walkthrough: a new AOT op

Assume you already have a working JIT override for the op, or at least a DSL
kernel you can compile from Python.

### Step 0: check the op is structured, and how it resolves

`ATEN_OP` must name exactly one structured group. Check
`aten/src/ATen/native/native_functions.yaml` for `structured: True` on the
`.out` variant. If the base name has several structured overloads, qualify it
(`"sum.dim_IntList"`, not `"sum"`). If the op is not structured, this system
cannot serve it -- structure it upstream first.

While you are there, find out whether META precomputes anything for the op:

```
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('g', 'tools/native_aot/gen_aot_lib.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
print('index_add       ->', g.precomputed_args('index_add'))
print('topk            ->', g.precomputed_args('topk'))
print('sum.dim_IntList ->', g.precomputed_args('sum.dim_IntList'))
"
```

### Step 1: add `build(spec)` to the kernel module

`build(spec)` takes one expanded grid point and returns the compile inputs plus
the marshalling metadata. For CuTeDSL that is the callable, the fake args, and
the tensor-arg descriptors:

```python
# torch/_native/ops/topk/cutedsl_kernels.py
def build(spec: dict) -> dict:
    """AOT builder: one manifest spec point -> compile inputs + sidecar.
    ...
    """
    dtype = _DTYPES[spec["dtype"]]
    N, K = int(spec["N"]), int(spec["K"])
    deterministic = bool(spec.get("deterministic", False))
    batch_sym = cute.sym_int()
    # ... fake tensors, div hints ...
    prefix = f"topk_radix_{_DTYPE_SHORT[spec['dtype']]}_n{N}_k{K}_{det_tag}"
    return {
        "prefix": prefix,
        "fn": _RadixSelectTopK(
            N, K, deterministic=deterministic, in_dtype=dtype, index_dtype=Int64
        ),
        "fake_args": [x_fake, v_fake, i_fake, cute.runtime.make_fake_stream()],
        "tensor_args": [
            {
                "name": "mX",
                "dynamic_sizes": [0],
                "dynamic_strides": [0],
                "read_only": True,
            },
            {"name": "mValues", "dynamic_sizes": [0], "dynamic_strides": [0]},
            {"name": "mIndices", "dynamic_sizes": [0], "dynamic_strides": [0]},
        ],
    }
```

`prefix` names the artifacts and the exported C symbols, so it must be unique
across all declarations. Mark inputs `read_only` so the launcher marshals them
through `const_data_ptr()`; a mutable `data_ptr()` would materialize
copy-on-write inputs on every call.

Three more parts of the `tensor_args` contract are load-bearing, and each fails
late:

* `name` must match the DSL callable's **parameter** name. The launcher emits
  `<prefix>_Tensor_<name>_t <name>_s;` against the exported header's
  per-parameter struct typedefs (`toolchains.py`), so topk's `"mX"` exists only
  because `__call__(self, mX, mValues, mIndices, stream)` names it that. A
  renamed entry compiles the generated `.cpp` against a nonexistent type.
* `dynamic_sizes` / `dynamic_strides` are lists of tensor **dims**, and their
  order indexes the ABI's fixed-width slot arrays (`dynamic_shapes[slot] =
  <name>.size(dim)` against `int32_t dynamic_shapes[N]` in the header). The
  count must match the `cute.sym_int()`s the builder put in that fake tensor;
  too many overruns a C array.
* Non-tensor runtime arguments go in `scalar_args`:
  `[{"name": "N", "ctype": "int32_t"}]`. They become launcher parameters after
  the tensors and before the stream, in list order. This is the only way to pass
  a runtime scalar to a CuTeDSL kernel. topk needs none, so the marshalling is
  covered by the tools tests rather than by a shipped declaration.

Per-toolchain required keys are `REQUIRED_BUILD_KEYS` in
`tools/native_aot/toolchains.py`: `("fn", "fake_args", "tensor_args")` for
`cutedsl`, `("kernel_path", "kernel_name", "signature", "grid", "args")` for
`triton`. A missing key fails at export with a message naming it, e.g.
`cutedsl builder result for x is missing keys: ['fake_args']`.

### Step 2: write `aot.py`

Copy `ops/topk/aot.py` and adapt. Order of work that tends to go
smoothest:

1. `ATEN_OP` / `DISPATCH_KEY` / `KERNEL_MODULE`, and the shared tables you will
   interpolate into both the Python and C++ sides.
2. `kernel_precompile_grid()`. Every axis you intend to gate coverage on must
   appear here. If eligibility is a boolean (`"outer": True`, `"tma": True`),
   pin it `True` in the grid, or unrelated calls would match on dtype alone.

   On *which* dtypes and shapes belong on the grid, the design doc's working
   policy: key off inputs -- **definitely bf16, probably fp32, maybe fp16, no to
   other dtypes** (let those JIT) -- and scale breadth by how much the op
   matters (a `mm` warrants more dtypes than an RMSNorm). topk's grid follows
   it: `_DTYPES` is fp32/bf16 only. Grids are
   expected to change over releases; nothing outside this tree may depend on a
   given point being AOT'd.
3. `cpp_dispatch_prelude()` + `cpp_dispatch(spec)` + `cpp_launch(spec, fn)`:
   the truth about what launches.
4. `covered_axes()` mirroring the prelude's *decidable* conditions, cheaply.
5. `cpp_covers()` as the C++ port of 4, if the op has a JIT override.

### Step 3: export, generate, relink, install

Your op is brand new, so stage 1 has never seen it: run a full
`pip install -e . --no-build-isolation` once first. Declarations are
`CONFIGURE_DEPENDS` codegen inputs, so that is what re-runs torchgen to emit your
op's DispatchStub and its consultation line -- and, crucially, the stub lands in
`NativeAotStubs.cpp`, which compiles into **torch_cpu**, while stage 2 installs
`libtorch_cuda.so` only. `<op>_aot_stub` is defined in libtorch_cpu and
undefined-and-imported in libtorch_cuda, so stage 2 alone would install a
libtorch_cuda referencing a symbol the installed libtorch_cpu does not export.
Afterwards, kernel and hook edits need stage 2 only.

The short version of stage 2 is one command, and it is the one to use:

```
# export + generate + relink torch_cuda + copy into the installed torch/lib
python tools/native_aot/build_stage2.py
```

By hand it is four steps, not three:

```
python tools/native_aot/export.py --ops <ops-dir-name>
python tools/native_aot/gen_aot_lib.py
cmake --build build --target torch_cuda
cp build/lib/libtorch_cuda.so \
   "$(python -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("torch._C").origin))')/lib/"
```

The copy is not optional. `cmake --build` stops at
`build/lib/libtorch_cuda.so`; the *installed* library is a separate file (an
editable install serves Python from the source tree while the compiled artifacts
live where `torch._C` resolves to), so without the copy Step 4 probes the old
library and reports `embedded: False` or misses your new op. `build_stage2.py`
does exactly this copy, via `_installed_lib_dir()`.

Two more things the by-hand path assumes:

* `--ops` accepts either the `ops/<dir>` name or an `ATEN_OP`. Artifacts and the
  generated `.cpp` are named by **`decl_id`**, not by the directory:
  `ops/topk/aot.py` exports into `build/native_aot/topk/`, and a family module in
  `ops/<dir>/` declaring `sum.dim_IntList` and `amax` would export into
  `build/native_aot/sum_dim_IntList/` and `build/native_aot/amax/` -- one dir per
  declaration, never one per directory.
* `gen_aot_lib.py` is unfiltered and checks every artifact dir, so a filtered
  `export.py --ops <yours>` can leave it failing on somebody *else's* stale
  artifacts (see the staleness pitfall in
  [6.2](#62-pitfall-checklist)). If that happens, re-run `export.py` with no
  `--ops`.

Read the generated `build/native_aot/<decl_id>/aot_<decl_id>_cuda.cpp`. It is
the single best debugging artifact: the arch gate, your prelude verbatim, the
branch list, and (if any) the covers body, with the precompute note at the top.

### Step 4: verify routing

```
# from a directory that is NOT the repo root: `python -c` puts the cwd on
# sys.path, so the source torch/ would shadow the installed wheel
cd test && python -c "
import torch
from torch._native import _native_aot_embedded, aot_enabled
print('embedded:', _native_aot_embedded(), 'switch:', aot_enabled())
from torch._native import aot_manifest
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
print('covered:', aot_manifest.covers('topk', 'CUDA', (x, 64), {}))
"
```

Then confirm the kernel actually fires: profile the call and look for your
kernel name with the JIT layer masked (`TORCH_DISABLE_NATIVE_JIT=1`), so any DSL
kernel in the profile can only have come from the AOT hook. That is exactly the
technique the end-to-end tests use.

### Step 5: tests

Correctness of the numerics belongs in the op's existing OpInfo/op suites. What
is specific to this layer is *routing*: covered calls hit the AOT kernel,
uncovered-but-JIT-eligible calls hit the JIT override, and everything else lands
on stock aten. See `test/python_native/test_native_aot.py` (topk) and its
per-op siblings for the pattern, and `test/python_native/test_aot_manifest.py`
for coverage/switch behavior with fixture declarations.

---

## 6. Real-world patterns

### 6.1 Where the worked examples are

This stack lands one declaration, so the examples are section 4.4 (topk's
declaration, annotated) and section 5 (adding an op end to end). The patterns that
need other shapes of op -- a family generated from one table, precomputed dims, a
prelude that classifies with TensorIterator, Triton's specialization parity -- are
documented with the declarations that introduce them, not here.

### 6.2 Pitfall checklist

Every item here is a real trap recorded in the code or the design doc.

* **Order zero-checks before divisions in C++.** An integer division by zero in a
  prelude is a process-killing SIGFPE, not a catchable exception.
* **Know whether your `dim` arrives wrapped.** A raw negative dim compared
  against `self.dim() - 1` silently declines every `dim=-1` call to stock aten.
  The generated `.cpp` header states the answer per op.
* **Never call `data_ptr()` in `covered_axes`.** It materializes copy-on-write
  inputs, and coverage runs on every call. Use `storage_offset()`; re-check the
  pointer in C++. On the C++ side, mark inputs `read_only` in `tensor_args` so
  the launcher uses `const_data_ptr()`.
* **Doubled braces in C++-returning f-strings.** `self.view({{M, N}})`,
  `if (...) {{ ... }}`. Easy to get wrong when pasting real C++ in.
* **Coverage narrower than the stub is fine; wider is a bug.** Wider means a
  call gave up its JIT route and then got declined by the stub, landing on stock
  aten unaccelerated.
* **`cpp_covers` sees the functional schema.** It runs pre-`meta()`, so outputs
  do not exist; the out-variant outputs appear as trailing optionals. Do not
  write it against the impl signature.
* **`cpp_launch` must not allocate outputs and must not implement fallback.**
  Outputs come from `meta()`; the chain's `return false` is the only fallback. A
  scratch buffer is fine.
* **A list in the grid cross-multiplies; a tuple does not.** Use a tuple for a
  compound value that must stay one axis point.
* **Grid values must survive a JSON round trip.** The sidecar stores the spec,
  and skip detection compares across that round trip.
* **Prefixes must be globally unique.** Two declarations sharing one export
  duplicate C symbols and fail at link, far from the cause.
* **Declaration objects must be mutable.** The loader materializes
  `d.ARCHS = tuple(...)`; frozen dataclasses and `__slots__` classes fail.
* **Families return instances, not classes.** With plain methods the class
  itself fails arity checks because `self` counts as a positional parameter.
* **Positional parameters with defaults count toward arity.** Put extra state on
  the declaration object, not in an extra parameter.
* **Do not hand-write arch checks.** The gate is injected into both the stub and
  `cpp_covers`; a hand-written one can contradict it.
* **A `covered_axes` bug is silent.** Exceptions degrade to "uncovered", so a
  broken projection quietly disables AOT rather than raising. Call it directly
  when debugging.
* **`covered_axes` must bind out-variant calls too.** Every overload of the
  group (base, `.out`, in-place) resolves to one declaration, so a signature
  that omits the outputs raises `TypeError` on an `out=` call, which degrades to
  "uncovered" -- a silent loss of the AOT route. Take `out=None` or
  `*args, **kwargs`.
* **Re-export after editing a kernel.** Sidecars record a source-closure hash;
  `gen_aot_lib` refuses stale pairings. The closure over-approximates badly: it
  is every **loaded** `torch._native` and `torchgen.native_aot` module, and
  `ops/__init__.py` eagerly imports every op package, so the closure spans all
  of them (a topk sidecar records ~50 sources, including `registry.py`,
  `aot_manifest.py` and `ops/foreach_mm/impl.py`). Editing `export.py`,
  `toolchains.py`, `torchgen/native_aot_decl.py`,
  `torchgen/native_aot_spec_grid.py` or the router therefore invalidates
  **every** artifact and means a full re-export. Once a second declaration
  exists, expect your first `gen_aot_lib.py` run to fail on the other op
  (`RuntimeError: <op>: 2 artifact(s) were exported from different kernel
  sources than the current tree`); run `export.py` with no `--ops` first.
* **Re-run generation after renaming or removing a declaration.** Otherwise the
  orphaned generated `.cpp` is still globbed and references a stub that no
  longer exists. `gen_aot_lib` deletes those for you when it runs.
* **An artifact-free build makes routing tests pass vacuously.** Assert
  `_native_aot_embedded()` first.

---

## 7. Arch gating

Declarations never hand-write sm checks. `gen_aot_lib` emits the gate from the
intersection of what the declaration says it supports (`ARCHS`) and what was
actually compiled (the sidecars' `arch`), and injects it into both the stub and
`cpp_covers`:

```python
# tools/native_aot/gen_aot_lib.py
def _device_match(major: int, minor: int) -> str:
    return f"{_PROPS_LOCAL}->major == {major} && {_PROPS_LOCAL}->minor == {minor}"

# _by_arch groups the sidecars per compute capability (dropping the loser of the
# sm_100/sm_100a tie-break), and the gate accepts the OR over those groups:
#   const auto* _naot_props = at::cuda::getCurrentDeviceProperties();
#   if (!((_naot_props->major == 10 && _naot_props->minor == 0))) return false;
```

Two things follow. The gate matches the **full capability**, major and minor, so
`sm_100` and `sm_103` are distinct and a 10.3 device declines kernels compiled for
10.0 rather than loading them. And shipping an arch the declaration disowns is a
packaging bug rather than a wider gate: generation refuses it, names the arch trees
to delete, and says why a bare re-export will not clear them.

What the standard build actually exports is narrower than the default `ARCHS`:

```python
# tools/native_aot/export.py
# Arch allow-list for the AUTOMATIC export path: which entries of the main
# build's TORCH_CUDA_ARCH_LIST are ELIGIBLE for AOT kernels. A filter, never
# a build list -- it cannot cause an export, only permit one, and an explicit
# `--arch` bypasses it entirely.
# ... (see the file for the full rationale, including what admitting a
# capability costs)
EXPORTABLE_ARCHES = ("sm_90", "sm_90a", "sm_100", "sm_100a")
```

Both spellings of the same compute capability are listed because they are
distinct nvcc targets and different builds use different ones for the SAME
hardware: `10.0a` (arch-conditional, what `tcgen05`/`wgmma` need) in
`.github/workflows/b200-native-aot.yml`, plain `10.0` in every other Blackwell
job and in the shipped manywheel lists. Omitting either would make those builds
silently export nothing.

Hopper is in the list, so a builder with `TORCH_CUDA_ARCH_LIST=9.0a` exports
for it -- and past this filter the DSL runtimes are required, so such a build
FAILS rather than skipping when they are missing. A list naming only arches
outside the tuple (say `8.6`) still prints "nothing to export" and stage 2 skips:
no artifacts, no error, stock aten and JIT behavior. On-device export (unset
`TORCH_CUDA_ARCH_LIST` plus a local GPU) is filtered by the same tuple, since
build_stage2 reads it to decide whether stage 2 runs at all.

Multi-arch export (`--arch sm_90a sm_100a`) nests artifacts under
`<out-dir>/<arch>/<decl_id>/`, and that is the only layout -- a single-arch
export nests too, so adding an arch to an op is one more directory. Generation
groups the sidecars per capability and emits a gate branch for each, and the
CMake it writes lists every source and object explicitly, by absolute path, so a
multi-arch tree links like any other.

On an unsupported device with artifacts present, the gate returns `false`, the
stub declines, and the call runs stock aten. Two cases:

* A declaration whose `cpp_covers` is used keeps its JIT route, because the same
  gate is injected there: coverage declines on the same devices the stub does.
  topk is this case, so nothing in this stack is exposed to the one below.
* A declaration with a JIT override and **no** `cpp_covers` falls back to the
  Python `covered_axes`, into which nothing is injected -- so unless it checks
  the arch by hand, it has no arch check at all. On a wheel whose artifacts
  target 10.0 only, an sm_80/sm_89 call to such an op is "covered", declines
  every JIT cond, then gets declined by the stub, and lands on stock aten --
  exactly the coverage-wider-than-the-stub bug. A declaration with no JIT
  override is safe whatever it does, having no route to lose. If you write one in
  the exposed shape, either add `cpp_covers`, or hand-check the capability in
  `covered_axes` (`get_device_capability(self.device)[0] >= 9`).

---

## 8. Build and test commands

### Full stage 2

```
# export + generate + relink torch_cuda + copy into the installed torch/lib
python tools/native_aot/build_stage2.py

# CI variant: also patch the relinked library back into a built wheel
python tools/native_aot/build_stage2.py --wheel "$(echo dist/*.whl)"
```

Stage 2 is chained automatically by `spin develop` / `spin editable` /
`spin install`, and by `.ci/pytorch/build.sh` after `pip_install_whl`. After a
raw `pip install -e .` it does **not** run; invoke it yourself.

It skips (printing why, leaving a normal artifacts-free build) when:

```python
# tools/native_aot/build_stage2.py
  * TORCH_NATIVE_AOT=0 in the environment (explicit opt-out)
  * no CUDA build (USE_CUDA off / no nvcc toolchain in the build)
  * no toolchain targets this build's backend (Toolchain.BACKENDS); a
    ROCm build skips here today, and gains AOT support by adding a
    toolchain class rather than by editing this gate
  * CUDA older than _MIN_CUDA_MAJOR (13), or a CUDA version that cannot
    be determined from the CMake cache or the installed torch
  * TORCH_CUDA_ARCH_LIST contains no exportable arch (see
    export.EXPORTABLE_ARCHES); on-device export runs when the arch list is
    unset and a supported GPU is present
```

The CUDA 13 floor is a cost decision, not a capability one: CUDA 12 builds top
out at sm_90 (`.ci/manywheel/build_env_setup.py`'s arch table) and every 13.x
config builds sm_90 too, so a 12.x export is a strict subset of what the 13.x
wheels already ship. Because `.ci/pytorch/build.sh` calls `install_cutlass_dsl`
only when `--print-verdict` says `RUN`, the 12.x jobs skip the DSL wheel install
as well as the export.

Past those checks the DSL runtimes are **required, not optional**: a toolchain
that targets this backend was asked for declared kernels, so a missing runtime
fails the build rather than shipping a wheel that silently underperforms. Same
for any later failure -- "silently shipping a wheel without the kernels it was
asked to embed is worse than failing loudly." `TORCH_NATIVE_AOT=0` is the
supported way to build without the DSL wheels.

### The steps by hand

```
# from the repo root, in a venv with torch built (pip install -e .
# --no-build-isolation, which is what configures the CMake tree at ./build)
# and the DSL wheel active
python tools/native_aot/export.py [--out-dir build/native_aot] [--ops topk] \
                                  [--force] [--jobs 8] [--arch sm_90a sm_100a]
python tools/native_aot/gen_aot_lib.py [--artifacts-dir build/native_aot] [--allow-stale]
cmake --build build --target torch_cuda
cp build/lib/libtorch_cuda.so "$(python -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("torch._C").origin))')/lib/"
```

`./build` is scikit-build-core's `build-dir` setting from `pyproject.toml`, not a
path you chose; `build_stage2.py` hardcodes the same assumption and errors with
`expected relinked library at ...` if it is elsewhere.

**Stage 2's incremental relink and copy cover `libtorch_cuda.so` only.** A
brand-new *declaration* also adds a DispatchStub to the generated
`NativeAotStubs.cpp`, which compiles into **torch_cpu** (the `--target
torch_cuda` relink pulls torch_cpu in as a dependency, but stage 2 copies only
libtorch_cuda.so into the install). So the first build after adding a
declaration must be a full stage 1 (`pip install -e . --no-build-isolation`);
after that, stage 2 alone suffices for kernel and hook changes.

Notes:

* `--jobs` defaults to the torch build's own parallelism (`MAX_JOBS`, then
  `CMAKE_BUILD_PARALLEL_LEVEL`, then half the CPU count). `--jobs 1` forces
  serial. Compiles run on a single **forkserver** pool covering every
  `(point, arch)` job; plain `fork` would hand workers the parent's dead CUDA
  context.
* Any number of arches works at any `--jobs`, including `--jobs 1`: the arch
  is per-compile state, so one process can export for several arches.
* With an explicit `--arch`, export never touches the CUDA driver (CuTeDSL via
  the `--gpu-arch` compile option, Triton via a fixed-target driver), so
  kernels build on GPU-less machines. CuTeDSL needs one cheap kernel-free
  compile first to initialize its JIT engine, or `export_to_c` fails with
  `Failed to dump object file with PIC relocation`; see
  `tools/native_aot/cutedsl_warmup.py`.
* Export is idempotent: a point is skipped when a sidecar records the same spec
  **and** the recorded source closure still matches the tree. Edit a kernel and
  the affected points re-export without `--force`.
* `gen_aot_lib` refuses stale sidecars rather than emitting a garbled `.cpp`:
  re-run export, or pass `--allow-stale` knowingly.

### Tests

```
# routing / coverage / switch behavior (needs CUDA; AOT-specific tests skip
# without embedded artifacts)
python test/run_test.py --include python_native/test_native_aot \
    python_native/test_aot_manifest

# or directly
python test/python_native/test_aot_manifest.py

# tool-level tests: no GPU, no built torch needed (this is what CI lint runs)
PYTHONPATH=$(pwd) pytest tools/test/test_native_aot_tools.py tools/test/test_native_aot.py \
    -o "python_files=test*.py"
```

`tools/test/test_native_aot.py` covers the torchgen side against a hand-written
`native_functions` fixture (golden stub declarations/definitions, the exact
consultation line for the functional and `out=` variants, and that a wrapper
with no manifest is byte-identical to before).
`tools/test/test_native_aot_tools.py` covers grid expansion, arch handling,
staleness, codegen shapes, and wheel patching.

The CI check that the wheel under test actually carries kernels, worth running
locally when a test "passes" suspiciously:

```
# from a directory that is NOT the repo root (the source torch/ would shadow
# the installed wheel)
cd test && python -c "
from torch._native import _native_aot_embedded
assert _native_aot_embedded(), 'AOT kernels not embedded: stage 2 did not run in the build'
print('native-AOT: embedded kernels detected')
"
```

The dedicated workflow is `.github/workflows/b200-native-aot.yml`: builds with
`cuda-arch-list: '10.0a'` and runs the `native_aot` test config on
`linux.dgx.b200`, on PRs touching the native-AOT paths (`tools/native_aot/**`,
`torch/_native/**`, `torchgen/native_aot*.py`, `test/python_native/**`, the
stage-2 build shells and the workflow itself), nightly, and on
`ciflow/b200-native-aot/*` tags.

---

## 9. Runtime switches

| switch | effect |
| ------ | ------ |
| `torch._native._native_aot_embedded()` | True iff this `libtorch_cuda` was linked with AOT artifacts. Detected by probing for any registered `_native_aot::` schema; cached; initializes no CUDA. |
| `torch._native.aot_enabled()` / `set_aot_enabled(bool)` | read/write `at::globalContext().allowNativeAot()`, the switch every stub consultation checks. False gives stock-aten behavior even with the library loaded. |
| `TORCH_DISABLE_NATIVE_AOT=1` | at `torch._native` import, flips that switch off -- but only on an embedded build (there is nothing to mask otherwise). |
| `TORCH_DISABLE_NATIVE_JIT=1` | masks the Python JIT override layer (pre-existing switch). Useful to prove a kernel came from the AOT hook. |
| `torch.backends.python_native.<dsl>.disabled()` | context manager masking **both** layers, and restoring the previous AOT state in a `finally`. This is the correct way to compute a stock-aten reference. |
| `TORCH_NATIVE_AOT=0` | build-time: skip stage 2 entirely (documented in CONTRIBUTING.md with the other build switches). |
| `NATIVE_AOT_ARTIFACTS_DIR` | CMake cache path for the artifact tree (default `${CMAKE_BINARY_DIR}/native_aot`). |

Two distinct "off" states behave differently, on purpose:

* **Artifacts absent** (stage 2 never ran): coverage still declines the JIT
  route, the stub has no kernel, and covered calls land on **stock aten**.
  Correct but unaccelerated. Artifact presence is a static property of the
  process, deliberately not probed per call.
* **Switch off** (`set_aot_enabled(False)` / `TORCH_DISABLE_NATIVE_AOT=1`):
  coverage itself returns `False`, so covered calls **keep** their JIT route.
  Masking must not lose both accelerated routes.

The C++ side:

```cpp
// aten/src/ATen/Context.h
  // Gates the native-AOT DispatchStubs consulted by generated structured
  // wrappers (see NativeAotStubs.h). Off means stock kernels even with an
  // AOT kernel library loaded, e.g. for reference computations.
  bool allowNativeAot() const;
  void setAllowNativeAot(bool /*b*/);
```

```cpp
// aten/src/ATen/Context.cpp
bool Context::allowNativeAot() const {
  // Relaxed: an independent on/off flag consulted per op call; no data
  // is published under it (the stub registration has its own fences).
  return allow_native_aot.load(std::memory_order_relaxed);
}
```

Default is on (`std::atomic<bool> allow_native_aot{true}`). The switch alone is
inert: the actual opt-in is whether the separately-built AOT library was linked.
The design doc left "should we be able to disable AOT kernels at runtime?" open
and voted no; the implementation shipped the switch anyway, on that reasoning
plus the need for stock-aten references in tests.

---

## 10. Limitations and restrictions

* **The op must be structured.** The consultation lives between `meta()` and
  `impl()`. Unstructured ops are served by the JIT layer only; the prescribed
  remedy is to structure the op upstream first (`var.correction` is named as a
  candidate in the validator's error text). Composite ops (`relu`) and
  multi-output ops with mixed out dtypes (`frexp.Tensor`) are out for the same
  reason.
* **`ATEN_OP` must resolve to exactly one structured group.** Ambiguous base
  names are a hard codegen error, not a silent pick.
* **Only Hopper and Blackwell export in the standard build.**
  `EXPORTABLE_ARCHES` (in `torchgen/native_aot_decl.py`, beside `KNOWN_ARCHES`) is
  `("sm_90", "sm_90a", "sm_100", "sm_100a")`. Other arches in
  `TORCH_CUDA_ARCH_LIST` are skipped, not failed, so those builds ship without
  artifacts. Both admitted capabilities ship where a build's arch list names them:
  a release CUDA 13.x wheel carries topk for `sm_90` and `sm_100` alike. No CI job
  exercises AOT kernels on Hopper yet -- only `b200-native-aot.yml` runs them -- so
  Hopper's coverage is the tools suite plus the Blackwell job.
* **A capability is admitted under one spelling.** `TORCH_CUDA_ARCH_LIST` naming
  both `10.0` and `10.0a` collapses to the arch-conditional one, so a declaration
  pinning only the plain spelling exports nothing for that build and says so.
* **DSL runtimes are not standard torch build dependencies.** `nvidia_cutlass_dsl`
  and `tvm_ffi` (and Triton for triton-kind ops) must be present for stage 2 to
  do anything. The design doc records this as an implication for release
  builders, not just developers.
* **Eligibility is stated up to three times and kept in sync by hand**:
  `covered_axes`, `cpp_covers`, and `cpp_dispatch_prelude` + `cpp_dispatch`.
  Nothing checks agreement. Drift is benign (both sides decline -> stock aten)
  but wasteful, and coverage wider than the stub silently costs the JIT route.
* **Coverage does not check that a kernel exists for the point.** It matches
  against the declaration's grid, not against the shipped sidecars. A point the
  export skipped is still "covered", and such a call declines JIT and lands on
  stock aten.
* **Kernels are linked into `libtorch_cuda.so`.** Whether they should live in a
  separate library is an open question in the design doc.
* **`cpp_helpers` has no in-tree user.** It is validated and emitted, but
  untested by example.

### Known future work

From the design doc, still open:

* Move the coverage short-circuit ahead of the Python dispatcher interpose (the
  "parked C++ router-prefix idea"). That would let a covered call run the pure
  stub chain, i.e. at or below aten host cost. The doc's companion item, writing
  `cpp_covers` wherever an op also has a JIT route, is settled: the contract
  supports it and topk uses it.
* A documented way for users to pre-JIT the kernels they care about but that are
  not on an AOT grid, and publishing the dtype/shape policy (stated for authors
  in [Step 2](#step-2-write-aotpy)) in user-facing docs, along with the fact
  that grids can change between releases.
* One-stage vs two-stage build. One stage "feels cleanest" per the doc, but the
  restrictions it needs (kernels structured as pure DSL, declaration and kernel
  code stdlib-only) were relaxed for the pointwise and reduction kernel modules;
  going back would be a large amount of op-side work.
* Tooling to author declarations: the design doc floats an AI skill for this
  workflow (structured-op check, `build(spec)`, `aot.py`, export/relink, routing
  test). Section 5 plus the pitfall checklist is the manual version of it. The
  mechanical parts (arity/cardinality, doubled braces, prefix uniqueness,
  grid/coverage mirroring) are exactly what such a tool would enforce up front
  rather than at load or link time.

---

## 11. Where the code lives

Build tooling (`tools/native_aot/`):

| file | role |
| ---- | ---- |
| `decl.py` | the declaration contract you program against (module docstring). Re-exports the mechanism from `torchgen/native_aot_decl.py`, which is where the `AotDeclaration` protocol, `decl_id`, and the validating loader / discovery actually live: the wheel ships `torchgen` but not `tools/`, and installed torchgen must load declarations out of tree. |
| `export.py` | stage-2 export driver: grid fan-out, one forkserver pool over all `(point, arch)` jobs, per-point compile via the toolchain, sidecars, staleness (`sources_current`), sidecar integrity, `EXPORTABLE_ARCHES`, `archs_from_cuda_arch_list`. |
| `cutedsl_warmup.py` | one kernel-free `cute.compile` per process, so `export_to_c` works for any `--gpu-arch` (its own module: `from __future__ import annotations` would stringify the jit annotation). |
| `toolchains.py` | one class per DSL: build-result validation, compile/export, launcher codegen; `TOOLCHAINS` registry. |
| `gen_aot_lib.py` | declarations + sidecars -> `aot_<decl_id>_<key>.cpp`: arch gate, prelude, dispatch chain, stub registration, covers op; `precomputed_args`, `covers_signature`, orphan cleanup. |
| `build_stage2.py` | the driver: skip ladder, export, generate, targeted `torch_cuda` relink, copy into the installed torch, optional wheel patch. |

Codegen (stage 1):

| file | role |
| ---- | ---- |
| `torchgen/native_aot.py` | discovery + validation (`parse_native_aot_manifests`, `validate_native_aot_manifests`) and the three emitters (`gen_stub_declaration`, `gen_stub_definition`, `gen_stub_consultation`). |
| `torchgen/native_aot_decl.py` | the declaration mechanism: `AotDeclaration`, `decl_id_for_op`/`decl_id`, `load_by_path`, `load_declarations`, `discover_declarations`. Surfaced to authors as `tools/native_aot/decl.py`. |
| `torchgen/native_aot_spec_grid.py` | `expand_specs`: the grid cross-product, shared by export and runtime coverage. |
| `torchgen/gen.py` | wiring: the `--native-aot-ops-dir` flag, per-key manifest filtering, and writing `NativeAotStubs.{h,cpp}`. |
| `torchgen/dest/register_dispatch_key.py` | the call site: replaces `op.impl(...)` with the consultation in the structured wrapper, and adds the stubs include for CUDA. |
| `aten/src/ATen/templates/NativeAotStubs.{h,cpp}` | the stub header/source templates. |
| `cmake/Codegen.cmake` | declarations as codegen inputs (`CONFIGURE_DEPENDS`). |
| `caffe2/CMakeLists.txt` | one `include(... OPTIONAL)` of the generated `native_aot.cmake`; no logic. |

Runtime:

| file | role |
| ---- | ---- |
| `torch/_native/aot_manifest.py` | coverage objects, `covers()`, the `cpp_covers` fast path, `get_coverage` symbol resolution. |
| `torch/_native/registry.py` | the router's once-per-call coverage check. |
| `torch/_native/__init__.py` | `_native_aot_embedded()`, `aot_enabled()` / `set_aot_enabled()`, `TORCH_DISABLE_NATIVE_AOT`. |
| `aten/src/ATen/Context.{h,cpp}` | `allowNativeAot()` / `setAllowNativeAot()`. |
| `torch/csrc/Module.cpp` | `_get_native_aot_enabled` / `_set_native_aot_enabled` bindings. |
| `torch/backends/python_native/__init__.py` | `disabled()`, which masks both layers. |

Declarations (`torch/_native/ops/<op>/aot.py`), in rough order of difficulty:

| declaration | what it demonstrates |
| ----------- | -------------------- |
| `topk/aot.py` | the only declaration in this stack: a complete single-op declaration, with determinism as a grid axis and `cpp_covers` for the router's fast path. |

Tests: `test/python_native/test_aot_manifest.py`,
`test/python_native/test_native_aot*.py`, `tools/test/test_native_aot.py`,
`tools/test/test_native_aot_tools.py`.
