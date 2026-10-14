# Custom CUDA LDL kernel — session notes

Handoff notes for `torch.linalg.ldl_factor(x, hermitian=True)` on the
`nikitaved/ldl_factor` branch. Written to be picked up on another machine.

**Next session: start with the TODO list in section 10.**

Last updated after the second round of work (batching moved into the kernel
file, `n = 0/1`, the last-column fix, the cooperative-panel tests, the standalone
kernel harness, and the LAPACK-style accuracy tests). Recent history:

```
f683b820c84 more tests
8409d0b5806 more tests
69b16d89951 more tests and bug fix with the last diagonal entry
698910054a2 more tests
b8c6f4a1dd7 add more tests
21672c07138 skip -> more enabled tests
2b8210c2b3f add support for n = 0/1, move batche support into the kernel
```

`test_ldl_custom.py` may carry uncommitted edits on top of this (the switch to
the LAPACK residual, section 7).

---

## 1. READ THIS FIRST — how to actually reach the kernel

`ldl_factor_kernel` (`aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp`)
sends `hermitian=True` to the custom kernel under the **default** and the
**cusolver** backends; the `magma` backend goes to MAGMA and `hermitian=False`
(real symmetric or complex symmetric) to cuSOLVER (MAGMA when built without it):

```python
torch.linalg.ldl_factor_ex(A, hermitian=True)              # hermitian=True REQUIRED
```

The default routing was added on 2026-10-14 (section 5). Before it,
the default backend sent `complex && hermitian` to **MAGMA** and everything else
to **cuSOLVER**, so the kernel was reached only with
`preferred_linalg_library("cusolver")`. That cost hours in the first session: a
full green test suite was reported as validating the kernel when it was in fact
exercising MAGMA. `test_ldl_custom.py` still pins `cusolver` in `setUp()`, so it
reaches the kernel either way, and `test_dispatch` checks every route with the
profiler (kernel name `ldl_diagonal_panel_fused_kernel`).

The gate is on `hermitian` only; **real dtypes reach the kernel too**.

The kernel and everything around it is in
`aten/src/ATen/native/cuda/linalg/LURecBlas3Kernel.cu`
(`ldl_factor_blas3_kernel` is the entry point; the panel kernel is
`ldl_diagonal_panel_fused_kernel`).

---

## 2. Environment & build

```bash
eval "$(micromamba shell hook --shell bash)" && micromamba activate kernel
cd /home/nik/pytorch
python setup.py develop          # what was used in these sessions
```

Note `CLAUDE.md` says to build only via `pip install -e . -v --no-build-isolation`;
`setup.py develop` was used at the user's explicit instruction.

- Incremental rebuild of the kernel file is ~5 ninja steps, under a minute or two.
- The kernel lives in **`libtorch_cuda_linalg.so`** (`build/lib/`), not
  `libtorch_cuda.so`. Check that one's timestamp when verifying a rebuild landed.
- If a rebuild appears to do nothing, `touch` the `.cu`.
- Hardware used: 8x H100 (sm_90). `build/CMakeCache.txt` shows
  `CMAKE_CUDA_ARCHITECTURES=75`, which is a **red herring**;
  `torch.cuda.get_arch_list()` returns `['sm_90']`, which is the real answer.
- `compute-sanitizer` is at `/usr/local/cuda/bin/`.
- Later also run on 4x GB200 (sm_100, aarch64 Grace, CUDA 13.4). pytest is not
  installed in the `kernel` env there; use `python test_ldl_custom.py [-k ...]`.
  The full suite takes ~42 minutes there (vs ~7 on the H100 box).

---

## 3. The main bug and its fix (done)

### Symptom
`test.py` failed at `n=33`, complex128, with round-trip error `0.0973` on a
matrix with `cond ≈ 2.8`. Positive-definite input passed; `n ≤ 32` passed.

### Root cause
The panel kernel defers the trailing block `S = dLD[panel_end:, panel_end:]` to
the end-of-panel GEMM, updating only two regions in-panel. But Bunch-Kaufman
scans the *whole* column and can select a pivot inside `S`, where values are
stale. Consequences, all from the one cause:

1. `sigma` and `a_rr` are computed from stale data → wrong pivot decision.
2. The interchange mixes a current row/col with a stale one.
3. The deferred GEMM then applies its update to the wrong entries.

PD input passes because it never interchanges; `n ≤ 32` passes because
`panel_end == n` leaves `S` empty. **The first wrong column is always exactly
the first out-of-panel pivot** — verified across many sizes.

### Fix (user's design, implemented)
When `ilambda >= panel_end` and an update is pending, bring that row/col up to
date **in place before the interchange**, keeping the correction in scratch, and
undo it after the swap. Doing it before the swap is what makes it clean: the
swap then carries the corrected values (diagonal included) to where they belong,
so no corner cases are needed.

Key detail: scratch holds the **correction**, not the old values. That makes
both branches the identical `+= corr` — rejecting restores the stale value,
accepting staleifies the row/col the swap pushed into `S`. Storing old values
would need two different fix-ups.

Why the same `corr` works for a different value after the swap: the pending
"debt" is attached to the *index* via the multipliers stored at columns
`t < curr_step`, and the swap only touches indices `>= curr_step`. So the swap
moves values but not the debt. (This is also why a moved column is updated with
the factors of its *new* position — see the zero-column discussion in section 5.)

---

## 4. Other fixes

**Dropped the forced-real `D`** (`42e254f7b55`). The kernel used to force
`Im(D11)=0`, `Im(D22)=0` and take `real()` of `D11`/`det`. In the in-place scheme
`U12` is an independently stored upper triangle, so realifying `D` desynced it
from the `U12` the trailing update consumes. Cost ~6 orders of magnitude.
LAPACK *does* realify — it's wrong **here** specifically, so don't "fix" it back.
(Diagonal entries are still made real where they are *read as a column*, in
`update_rowcol`, which is a different thing: see "last column" below.)

**Barriers** (`459a7097d88`). Three `__syncthreads()` added: before the `sigma`
search, after the `if (tid == 0)` pivot/info block, and at the end of the loop
body.

**`info` for 2×2 pivots.** The old test `dLD[piv,piv] == 0` is the *normal* state
for a 2×2 block. Now split by rank where `D` is known: `|D11| == 0` for 1×1,
`|det(D)| == 0` for 2×2.

**Illegal memory access.** `AGGREGATE_ARGMAX` tests `other_val > val`, false for
NaN, so an all-NaN column leaves `my_idx = -1`; `LinOff(t, -1, lda)` then reads
off the front of the buffer. Guarded with `ilambda < 0` falling back to a 1×1
pivot.

**Batching — now inside the kernel file.** The panel kernels run on a 1×1×1
grid and a single `dLD` pointer, so batching is a host loop. It used to live in
`BatchLinearAlgebra.cpp`; it is now in `ldl_factor_blas3_kernel`
(`LURecBlas3Kernel.cu`), which loops over `ldl_factor_blas3_single`. It uses
`view`, not `reshape`: `reshape` would silently copy on an awkward layout and the
in-place result would be discarded; `view` throws instead.

**Mirroring the lower triangle (memory).** The upper triangle of `LD` is filled
once for the whole batch in `ldl_factor_blas3_kernel`, before the per-matrix
loop. That is one launch instead of B, but the temporary now spans the whole batch.
Peak device memory of `ldl_factor_ex` is the input plus 2 copies (`LD` and one
temporary) for every dtype. Three traps on the way there:
- `LD.add_(LD.mT())` raises for real dtypes. TensorIterator's
  `assert_no_partial_overlap` rejects a self-aliasing transposed input, hence
  the `tril(-1)` copy.
- `LD.add_(LD.tril(-1).mH())` costs a second temporary for complex, because `add_`
  resolves the lazy conj view into a fresh copy. `conj()` before `tril` is no
  better: `tril` resolves the conj input into a copy first. The fix is
  `conj_physical_()` in place on the `tril` copy. Before the fix, complex128 at
  n=56000 ran out of memory on a 184 GiB GB200.
- The `tril` copy is scoped in a block. As a named local it stayed alive through
  the whole factorization and stacked with cuBLAS's 32 MiB GEMM workspace.

**`n = 0` and `n = 1`** are shortcuts in the same function: empty input returns;
`n == 1` sets pivots to 1 and `info = (LD == 0)`. (`n = 1` used to return
uninitialized pivots.)

**The last column (found while adding tests).** Two defects, one cause: the
outer loop was `while (step < n - 1)`, so no panel ever treated column `n-1` as
the current column, and it is `update_rowcol` that makes a Hermitian diagonal
real and the panel loop that checks `D == 0`. Result:
1. A zero **last pivot never set `info`** (`diag(1..n-1, 0)` gave 0, CPU gives `n`).
2. An imaginary part on the Hermitian diagonal leaked into the last diagonal
   entry of `LD`.

Fix, in the kernel: the host loop is now `while (step < n)`. When a panel
reaches `curr_step == n - 1` it handles the last column as a 1×1 pivot of its
own — sets `dipiv[n-1] = n`, sets `info = n` if `|D| == 0` and `info` is still 0,
and advances `curr_step` to `n`, which ends the loop. If the previous panel
stopped at `n-1` (the usual case: the trailing GEMM still has to run), the next
iteration launches a width-1 panel that does exactly this, *after* the GEMM.
Doing it at the previous kernel's exit does not work: the entry is stale there
and the GEMM changes it again. (An earlier attempt put it at the exit and left
rounding noise in the imaginary part at n=65; a host-side tensor-op version
worked but was replaced.)

**Cooperative multi-block panels (commit `a986db8fb6c`, tests added later).**
For more than `COOP_MIN_ROWS = 2048` remaining rows the panel kernel runs as a
cooperative grid (one row per thread, grid-wide argmax via double-buffered
partials), capped by `max_cooperative_panel_blocks()`; if the launch is too
large it falls back to the single-block kernel. On an H100 the kernel uses 48
registers at 256 threads, i.e. 5 blocks/SM, so the cap is 660 blocks and binds
only past n ≈ 169k — unreachable in a test (a float64 matrix that size is
~228 GB). See the standalone harness in section 7.

---

## 5. Open issues

**Zero row/column and `info` — NOT a bug (earlier notes had this wrong).**
CPU LAPACK and cuSOLVER report `info > 0` for an exactly zero row/column (every
case in a 36-case sweep matched CPU exactly, index included). The custom kernel
sometimes returns `info == 0` for the same input when the column is moved across
a panel boundary with an update pending. Reason: interchanges move data but not
`Lprev/Uprev`, so a moved column is updated with the factors of its *new*
position, and the exact zero survives only if the undo and the trailing GEMM
cancel bit for bit, which they do not (~1e-17 is left where `0.0` belongs).
LAPACK keeps the zero exact because its factors travel with the data. The
contract is only "`info` is set when a computed `D` entry is exactly zero", and
for a general singular matrix rounding decides that on every backend, so this is
a difference in a niche structured case, not a defect. Decided not to fix it
(detecting it at undo time would only catch this one structure, costs a scan of
the active column, and would report an index that is not LAPACK's). A comment
at the `info` update in the kernel records this; there is deliberately **no
test** that requires it. `test_zero_row_col_completes` requires only completion
and an in-range `info`.

**Dispatch. Done (2026-10-14).** `hermitian=True` goes to the custom kernel under
both the default and the `cusolver` backends, `magma` still goes to MAGMA, and
`hermitian=False` to cuSOLVER (MAGMA without it). The default case used to send
complex `hermitian=True` to MAGMA 2.5.4+; that block is gone. Both routes call
`ldl_factor_blas3_kernel(LD, pivots, info, hermitian)` identically, so the suite
is not run twice: it stays pinned to `cusolver`, and `test_dispatch` checks all
routes (default/cusolver/magma x hermitian x dtype) for the kernel name in the
profiler. (The odd brace formatting in that switch is upstream's.)

**`check_errors=True` on singular input** raises
`INTERNAL ASSERT FAILED ... Unknown error code: 1` from
`aten/src/ATen/native/BatchLinearAlgebra.cpp:1679`. **Reproduces on CPU** — a
pre-existing gap in the error mapping, unrelated to this kernel. Not pinned by a
test (it is a bug, not a contract). **Already known upstream, nothing posted:**
issue pytorch/pytorch#83624 (open, "Unhelpful error message from
torch.linalg.ldl_factor") and draft PR #197234 (open, "Fix INTERNAL ASSERT in
_linalg_check_errors for ldl_factor"). Related: #172825 (closed draft, "[MAGMA]
ldl_factor: deprecate magma backend and dispatch to cusolver unconditionally").

**Not testable on either machine:** the cooperative cap and the
`cudaErrorCooperativeLaunchTooLarge` fallback in the real library (see section
4; the standalone harness tests the kernel on small grids instead).
- On GB200 the cap is 760 blocks for float32/float64 (48 registers, 5 blocks/SM,
  152 SMs), 608 for complex64 (64 regs) and 456 for complex128 (76 regs). It binds
  past n = 194,560 / 155,648 / 116,736, where one copy of the matrix is already
  151-303 GB.
- The fallback could be forced with a green context with few SMs
  (`torch.cuda.green_contexts`). That needs `cuda.bindings`, which is not installed
  in the `kernel` env on either machine (H100 checked 2026-10-14). GB200 itself (sm_100, its panel widths) is now covered by the
  full suite.

**Resolved since the first notes:** `n = 1`; batching location; the "accuracy gap
vs CPU grows with n" (the first notes measured CUDA/CPU backward-error ratios of
3.6 to 76 for float64 and 1.9 to 180 for complex128; measured again after the
later changes, CUDA and CPU agree within a few percent on every structure, see
section 7).

---

## 6. The `W` scheme (prototyped, not implemented in CUDA)

`agent_space/wpanel.py` / `herm.py`. Mirrors LAPACK `zlahef`: `A`'s trailing
block receives **nothing** during the panel, panel columns are computed on demand
into `W`, and `Lw = W·D⁻¹` and `W` carry the row permutations while `A`'s output
`L` does not. Validated: pivots identical to LAPACK at every size and panel
width, backward error `3.45e-14` at n=512 vs LAPACK's `3.84e-14` — better than
the unblocked reference. Accuracy is **independent of `nb`**, so one rank-`nb`
GEMM per panel is preserved.

Because the trailing block is never partially updated, it structurally removes:
the staleness bug, the undo/GEMM cancellation problem (which would also make a
moved zero column exact), and the accuracy gap. Cost: a rewrite of
`ldl_diagonal_panel_fused_kernel` plus two `n × nb` workspaces (~10 MB at
n=10000, nb=32, complex128). With the accuracy gap now measured as closed in the
in-place kernel, the remaining motivation is structural, not numerical.

One trap found the hard way: in the `W` formulation the 2×2 `D` **must** be
symmetrized (`D[0,1] = conj(D[1,0])`). Without it, n=512 backward error is
`3.09e-07`; with it, `3.45e-14`. This is the opposite of the in-place scheme,
where forcing `D` real is harmful.

---

## 7. Tests

`test_ldl_custom.py` at repo root — **tracked** (force-added; `.gitignore` has
`/test_*.py`, so re-adding needs `git add -f`). About 10.9k test instances;
~7 minutes for everything (~2.3k are skipped on the CPU instance because they are
`@onlyCUDA`, or are structurally inapplicable).

```bash
./run_ldl_tests.sh                 # builds the standalone extension, runs everything
SANITIZE=1 ./run_ldl_tests.sh      # ... plus compute-sanitizer racecheck/synccheck
./run_ldl_panel_standalone.sh      # just the standalone-kernel tests (SANITIZE=1 works)
python test_ldl_custom.py -k determinism
```

All of them need an environment where this checkout's torch imports
(`micromamba activate kernel`). The suite forces the cusolver backend in `setUp`
and restores it in `tearDown`.

### What is covered
- 17 matrix structures; shapes 1→4097 (dense around the 32/64/512 boundaries);
  `n = 0/1` with all batch shapes incl. empty batch dims.
- All 230 compositions of `n` into 1s and 2s for n=2..10, with the achieved pivot
  layout **asserted**; ~90% 2×2 configurations at n=150..513.
- Batch shapes through `(2,2,2)`, non-contiguous batches (strided, permuted,
  expanded with stride 0), lazy conj/neg views (`test_lazy_views`, bitwise equal
  to the resolved input), batched `lda > n`, `out=`, `nrhs`, non-default stream,
  input on another GPU,
  CUDA-graph capture (must fail cleanly or reproduce eager), determinism (also
  under concurrent GPU load), half/bfloat16 (`NotImplementedError`).
- NaN/Inf input (column, single entry, all) through the real library.
- Only the lower triangle is read (junk in the upper triangle and in the
  imaginary part of a Hermitian diagonal must not matter), and a Hermitian `D`
  has a real diagonal.
- **Pivot encoding:** every test that checks pivots (`_check` and the singular /
  non-finite / cabs1 tests) uses `_pivot_encoding_error`, the full LAPACK lower
  encoding (1x1: `k + 1 <= p <= n`; 2x2: equal `-p` pair with `p >= k + 2`), not
  just the range.
- **Upper triangle of the output.** The front-end `tril_out`s the input into `LD`
  before dispatch, so every backend starts from zeros there. LAPACK and cuSOLVER
  leave them; MAGMA writes there from n ~ 300 on; the custom kernel leaves
  working values there at every n (max |upper| 3.5 at n = 5, 48 at n = 300).
  Decided (2026-10-14): leave it, since `ldl_solve` reads only the lower triangle,
  which `test_solve_ignores_upper_triangle` pins (NaN in the upper triangle gives
  a bitwise identical solve, on CPU and on CUDA for real dtypes).
- **Last column:** zero last pivot sets `info = n` (sizes 2..4100, all dtypes);
  `info` keeps an earlier zero; tiny nonzero last pivot is not singular; per-batch
  `info`; layouts with a last 1×1 or 2×2 pivot and 2×2 blocks straddling every
  panel boundary (periods 16/32/64 = where the panels of the four dtypes end);
  and exactly-zero last pivots that are only reached through updates, including
  after planted interchanges, in-panel and out of panel (see below).
- **Cooperative path:** n around 2048 (single block → cooperative boundary) up to
  8193 and 40000 (`test_very_large`), all structures that stress the grid-wide
  argmax (ties across blocks, maximum in the last block), single precision,
  batched, determinism.
- **`test_huge`** (GB200-sized, guarded by `largeTensorTest`; 8 CUDA instances, ~10
  min). It uses `zero_diagonal`: a 2x2 pivot with an interchange at every step,
  taken from anywhere in the column.
  - `int32_offsets`: n = 46341, so n² > 2³¹, for all dtypes.
  - `blocks_per_sm`: n = 115000 / 80000 / 80000 / 56000 for float32 / float64 /
    complex64 / complex128, i.e. 450 / 313 / 313 / 219 cooperative blocks, ~3 / 2 /
    2 / 1.4 per SM.
  - The grid sizes were confirmed with a profiler trace
    (`agent_space/huge_probe.py`).
  - `integer_valued` was left out: the residual cannot tell whether ties were
    broken correctly, which is what determinism and the standalone harness check.
- `ldl_solve` on CUDA for real dtypes with the custom kernel's output (both
  `hermitian` flags).

### Accuracy metric
LAPACK's scaled solve residual, as in `xPOT02`/`xSYT02` (from memory, not checked
against the source): `||B - AX||_1 / (||A||_1 ||X||_1 eps)` with `eps =
finfo.eps / 2` (LAPACK's `dlamch('E')` counts rounding). LAPACK's own drivers use a
threshold of 30.

The bound is `max(30, 1.5 · ref)` (`_resid_threshold(A, B)`), where `ref` is the
residual of `torch.linalg.solve` (LU) on the same `A`, `B`, on `A`'s device. The
residual's absolute level is **machine dependent**, so a fixed formula in `n` does
not travel:
- On H100/x86 it grew like `0.5·sqrt(n)` on the hardest inputs (22 at 2049, 98 at
  40000), and the first version of the bound was `max(30, 0.7·sqrt(n))`.
- On GB200/aarch64 (CPU LAPACK is ATLAS with Eigen BLAS there) it is about 2x that
  for complex: 40-46 at 2049, 136-158 at 8193, 188 at 40000 for float64.
- That 2x shows on every solver, CPU LDL, CPU LU and cuSOLVER LU included, and on
  pure-CPU test instances that never touch CUDA. 34 tests failed on GB200 until
  the bound was made relative.

Against cuSOLVER LU the custom kernel measured at most 1.02x, and against CPU LDL at
most 1.09x. CPU LDL against CPU LU (the reference on CPU-device instances) measured
at most 1.2x. The exception is `integer_valued`: there, Bunch-Kaufman LDL sits about
1.3x above LU from n ≈ 8000 on (1.33 at 8193, 1.30 at 40000, 1.36 at 56000), and CPU
LAPACK's LDL does the same within 0.03. That is the algorithm, not the kernel, and
it leaves `test_cooperative_panel`'s integer_valued n=8193 case with ~13% headroom
under the 1.5 factor (`agent_space/ratio_trend.py`). On H100 (MKL) the same case is
at 0.5-0.6x LU (10.3, residual factor), so the 1.3x is machine-dependent. The bound still has teeth: multiplying `LD` by `1 + 30·eps·randn`
pushes the residual 5-10x over it at every size (`agent_space/bound_mutation.py`).
The earlier tolerances (`1000·n·eps` with max-norms, and `max(50·cpu, 200·n·eps)`)
were ad hoc and 3 orders of magnitude loose; they are gone.

### Standalone kernel harness
`ldl_panel_standalone.cu` (repo root, **not** under `aten/.../linalg/`, which
CMake globs) holds a **copy** of `ldl_diagonal_panel_fused_kernel` between
`BEGIN COPY`/`END COPY` markers, plus a launcher that takes the grid size, thread
count, cooperative flag and `hermitian`. It is JIT-built by the test with
`torch.utils.cpp_extension.load`. It exists because the library never launches
grids it doesn't need, so grid sizes below the row count (what the cap would
force) are otherwise unreachable. Tests: results bit-identical for every grid
(1, 2, 3, 7 blocks, 256- and 1024-thread single block), all dtypes, panel widths
8..200, to completion or for the first panels; planted ties across block
boundaries (`1` vs `1j`, since the pivot magnitude is `|re|+|im|`); NaN/Inf
columns; a race stress with concurrent matmuls; and full factorizations at every
panel width (the GB200 widths included) checked against the residual.

**The copy must be refreshed by hand when the kernel changes.**
`test_standalone_kernel_matches_original` compares the whitespace-normalized
kernel text of both files and fails when they differ. The harness driver raises if
a panel makes no progress, so a regression at the last column errors instead of
looping forever.

### Exact constructions
When a test needs an *exactly* zero pivot (`exact_last_zero_matrix`), the matrix is
built from small dyadic numbers (diagonal ±2^k, couplings 2^j·{±1,±i}, last entry =
the exact Schur complement), so every update is exact in any order and CPU and CUDA
must agree. An optional `swaps` argument plants interchanges (`d_a = 1/2, d_b = 4,
A[a,b] = 1`, `v_a = v_b/4`). A randomly permuted version was tried and
rejected: interchanges and 2×2 blocks make the arithmetic inexact and CPU and CUDA
then disagree in both directions.

### Two rules the suite encodes, both learned painfully
1. **Never validate by comparing pivots or `LD` against CPU.** Bunch-Kaufman
   decisions sit on thresholds, so a different summation order legitimately
   picks a different, equally valid pivot sequence. Validate with the solve
   residual. (Exception: `test_pivots_match_cpu` for n ≤ 40, where seeds are fixed
   and it has held; and the exact constructions above.)
2. **The solve must run on CPU** for complex, since `torch.linalg.ldl_solve`
   rejects complex with `hermitian=True` on CUDA — exactly the combination the
   kernel exists for.

Controls that matter: `positive_definite` and `diagonal_only` trigger **zero**
interchanges, so a broken interchange path cannot hide behind them.

Weak spots, deliberate: `arrow` yields only 1–2 interchanges; `tridiagonal`
structurally cannot produce an out-of-panel pivot.

`test.py` at root is the user's original scratch repro (n ∈ {37,111,247,1023}).
It does **not** set the cusolver backend, so by default it exercises MAGMA.

---

## 8. Python prototypes — `agent_space/` (gitignored, `.gitignore:306`)

These will **not** transfer with a git clone. Copy them separately if needed.
(`agent_space/` now also holds throwaway measurement scripts and build/test logs
from the second round; none of them matter.)

| file | what it is |
|---|---|
| `bisect.py` | unblocked reference + `W`-panel, with snapshots. Most others `exec` its header |
| `blocked.py` | blocked in-place model reproducing the original bug; flush variant |
| `replay.py` | design 1 of the replay (validated) |
| `design2.py` | design 2 — in-place before the swap, i.e. what the kernel now does |
| `design4.py` | the three-way bind: additive vs swap-`U12` vs swap-both |
| `trace_zero.py` | pinpoints which operation perturbs a zero column |
| `wpanel.py`, `herm.py` | `W` scheme + the `D`-symmetrization finding |
| `nbsweep.py`, `control.py`, `pd.py`, `cabs1.py` | supporting experiments |
| `resid_vs_ref.py`, `ratio_trend.py` | GB200: custom residual vs CPU LDL and cuSOLVER LU |
| `bound_mutation.py` | the residual bound still catches a 30·eps perturbation |
| `huge_probe.py` | per-phase time/memory of a `test_huge` case, cooperative grid sizes from a profiler trace |
| `nan_argmax.py` | single NaN in the argmax vs grid size (TODO 10.2, not run yet) |

**`cabs1`:** the reference implementation's `.abs()` is wrong for complex. LAPACK
and the kernel both use `cabs1 = |re| + |im|`. With `.abs()` the reference
diverges from LAPACK at column 0; with `cabs1` it matches bit-for-bit. Fix this
in any reference used as an oracle.

---

## 9. Methodology notes

Things that cost real time and would cost it again:

- **Check which backend you're testing.** Everything else is downstream of this.
- **Pivot equality is not a correctness criterion.** Use the solve residual.
- **Prototype in Python against LAPACK before writing CUDA.** Every design that
  was validated in `agent_space/` first worked; the one written straight into
  CUDA (design 2) had bugs that took a rebuild cycle each to find.
- **A passing test proves nothing until you verify it exercises the path.**
  Instrument for the path (out-of-panel pivot counts per structure; a profiler
  trace showing the cooperative kernel actually launches at n=2049 and not at
  2048). And **mutate**: revert the fix (or break the copy of the kernel), expect
  the new tests to fail, restore. One strengthened test still passed without the
  fix until it was fed an imaginary diagonal.
- **Verify the build actually relinked** before interpreting results.
- **Do not assert behaviour that only rounding decides** (info for a general
  singular matrix, the position of an exact zero). Either build an exact
  construction or assert only what is guaranteed. The zero-column `info`
  difference was first logged as a bug and then retracted after working out why.
- **Measure a metric at every size you assert it at.** The residual was "measured
  n-independent" up to n=2049 and then failed at 4100, 8193 and 40000.
- **Guard test drivers against non-progress** (a loop that waits for a kernel to
  advance will spin forever on a regression).
- **A background job and `pkill -f` share a process list with your own shell:**
  matching on a word that is in your own command line kills the shell.

---

## 10. TODO (next session)

State at the end of the GB200 session:
- **Committed:** `0bbcef0`. The full suite passes on it on GB200 (10928 tests, 52 min).
- **Committed later:** `5a3dd9c` (the `n == 1` fix, `test_n1`,
  `test_nonfinite_imag_diagonal_is_ignored`) and `f0e4a2b` (these TODOs).
- **Committed later still:** `e27fc3b`, `353ffe0` (`at::imag` instead of `.imag()`,
  10.1.2/10.1.3). The working tree was clean on the H100 box after that.
- **H100 session, 2026-10-14:** `8263c00` (10.1.1), `9a57b5f` and `c15de04` (10.2),
  then 10.3: default dispatch, the kernel NOTE on subnormal and near-overflow 2x2
  pivots, `test_cabs1_overflow`, `test_peak_memory`, `test_dispatch`.
- `unittest -k` with a `*` matches the full test id, so a pattern needs a leading
  `*`: `-k "*test_n1_*cuda*"`. Several `-k` flags are OR-ed.

### 10.1 Bugs, in order

1. **2x2 pivot forms `det(D)` directly. Fixed (H100 session, uncommitted at the
   time of writing).**
   - The fix has two parts. The 2x2 inverse now uses LAPACK's scaled form:
     `det(D) = d01·d10·(r0·r1 − 1)` with `r0 = d00/d01`, `r1 = d11/d10`, and the
     scales `t/d01`, `t/d10` are applied in the L21 loop. The Bunch-Kaufman pivot
     test also formed a product: `sigma·diag_abs >= ALPHA·lambda²` is now
     `diag_abs >= ALPHA·lambda·(lambda/sigma)`, as in LAPACK. That product was
     the source of the false `info` below: at 1e-165 both sides underflow to 0,
     and `0 >= 0` picks a 1x1 pivot on the zero diagonal.
   - A 2x2 block can no longer be singular: the pivot test bounds `|r0·r1|`
     below `2·ALPHA² ≈ 0.82` (cabs1 vs modulus gives the 2). `info` now comes
     from 1x1 pivots only, which matches LAPACK (a zero column).
   - `test_extreme_scale`: zero_diagonal and indefinite, n = 2/64/300, scales
     1e±{153,160,165,250} (f64) and 1e±{19,21,23,30} (f32). On the old build 110
     of the 192 CUDA subtests failed. Measured after the fix
     (n up to 2049, scales up to 1e±300): the residual is scale-independent and
     within noise of CPU LDL.
   - The original analysis, before the fix:
   - The panel kernel computed `det = D00·D11 − D01·D10` and divided by it.
     Past ~sqrt(finfo.max) the product overflows; below ~sqrt(finfo.tiny) it
     underflows.
   - Overflow gives `x / inf = 0`: finite, plausible-looking garbage with
     `info = 0`. Underflow gives the same, or `det == 0` and a false `info > 0`.
   - Measured on `zero_diagonal`, n=64, against CPU LAPACK (residual ~1 at every
     scale):

     | dtype | wrong, info=0 | false info=1 |
     |---|---|---|
     | float64 | scale ≥ 1e153 (resid 8e13-1e15) and 1e-160 to 1e-162 (resid 4e9-2e13) | scale 1e-165 |
     | float32 | scale ≥ 1e19 (resid 2e6) and 1e-21 (resid 10.8) | scale ≤ 1e-23 |

   - The existing `huge_scale`/`tiny_scale` (1e±30, 1e±15) are far inside the
     safe range, which is why nothing caught this.
2. **`n == 1` ignored the Hermitian imaginary part. Fixed (folded into 10.1.3)
   and verified.** Note: `5a3dd9c` alone does not compile; it calls `LD.imag()`,
   and `imag` is a free function only in ATen (`at::imag`).
   - For `[[1j]]` CUDA returned `LD = 1j, info = 0`; CPU LAPACK and MAGMA return
     `LD = 0, info = 1`.
   - The fix zeroed the imaginary part of `LD` in the `n == 1` shortcut. It is now
     covered by the general fix of 10.1.3, which runs before that shortcut.
3. **A non-finite `Im(diag)` poisoned the real part. Fixed and verified (all 96
   CUDA instances of the two tests below pass).**
   - LAPACK ignores the imaginary part of a Hermitian diagonal. The kernel used to
     drop it only when the column became current (`update_rowcol`).
   - Before that, the trailing GEMM computes `beta·C = (1+0i)·C`, and `0·inf = NaN`
     moves an Inf/NaN from the imaginary part into the real part.
   - `test_nonfinite_imag_diagonal_is_ignored` on the build without the fix: every
     CPU instance passes; CUDA fails for n = 65, 300, 4100 (and complex64 at 33).
     The damage starts exactly at the first panel end (column 32 for complex128,
     26 for complex64), with hundreds of non-finite real entries in `LD` and
     different pivots. Without a GEMM (n=2, complex128 n=33) nothing is wrong.
   - Fix: `at::imag(LD.diagonal(0, -2, -1)).zero_()` for complex Hermitian input at
     the top of `ldl_factor_blas3_kernel`, before the `n == 1` shortcut, which is
     back to `pivots.fill_(1)` + `info`.
   - After the rebuild, run
     `python test_ldl_custom.py -k "*test_nonfinite_imag_diagonal*cuda*" -k "*test_n1_*cuda*"`.
     The CPU instances take ~35 min (ATLAS) and already passed.
   - The realification inside the kernel (`update_rowcol`) stays: the GEMM
     updates themselves leave rounding-level imaginary parts on the diagonal.

### 10.2 Missing tests: non-finite input

`test_nonfinite_input` only checks no crash, pivots in range and `0 <= info <= n`.
`test_panel_nonfinite_column` only checks grid invariance for a full NaN/Inf column.
Missing:

- **Single NaN among finite candidates: grid dependence. Not a bug; test added.**
  The worry was that `AGGREGATE_ARGMAX` never replaces a NaN once it holds one, so
  the winner would depend on the reduction order. But a NaN never gets in: every
  argmax starts from the sentinel `-1`, and `NaN > -1` and `NaN == -1` are both
  false. The reduction runs over NaN-free values with a total order, so it does
  not depend on the order. NaNs are skipped, and `ilambda == -1` only when every
  candidate is NaN (already guarded).
  - Measured on the standalone copy, n=4100, all dtypes, every grid. The NaN was
    first in the column, in the maximum's own thread (256 and 1024 threads), in
    its block, after it, in the last row, or everywhere but the maximum, and in
    column 0 or the candidate's column, real or imaginary part. The planted
    maximum won every time.
  - This differs from LAPACK only when the NaN is the first candidate, which
    `i?amax` returns. That is fine: the result is NaN garbage either way.
  - `test_pivot_search_skips_nan` pins it. Mutation: an argmax that takes a NaN
    fails all 28 CUDA instances.
- **NaN propagation, positions, `info`, `ldl_solve`. Done:
  `test_nonfinite_propagates`** (n = 5/65/300/2100, all dtypes, 11 placements:
  NaN on the first, an interior and the last diagonal entry, off the diagonal,
  only in the trailing block, only in the imaginary part; `-inf` in the trailing
  block; `+inf` and `-inf` in one column; Inf as a 1x1 or 2x2 pivot; Inf on the
  last diagonal).
  - Propagation holds everywhere: the kernel's LD has a non-finite entry in every
    case, as do CPU (MKL), cuSOLVER `sytrf` and MAGMA.
  - **`info` contract (decided): 0 for non-finite input.** `info` flags an
    exactly zero pivot only, and NaN/Inf is not zero. "Compare with CPU" was not
    possible. The references set `info > 0` in most NaN cases and some Inf cases,
    but disagree with one another on whether and where. For example, f64 n=65
    with NaN at `[0,0]` gives MKL 65 and cuSOLVER 0. With `-inf` in the trailing
    block at n=300, MKL gives 0 and cuSOLVER 300. Causes: unblocked `?sytf2` flags
    a NaN diagonal (`DISNAN(ABSAKK)`) and blocked `?lasyf` does not. MAGMA uses a
    different Bunch-Kaufman growth constant, so its pivots, and the positions it
    reports, differ for that reason too. Do not use MAGMA as a pivot reference.
    The test also checks that `check_errors=True` does not raise.
  - Real dtypes: `ldl_solve` on CUDA accepts the non-finite LD without failing.
- **Batched isolation. Done:** `test_batched_elements_are_independent` got
  `poisoned`, which puts a NaN in batch element 0. The others must still match
  factoring them alone.
- **Non-finite junk in the upper triangle. Done:**
  `test_only_lower_triangle_is_read` got `fill` = random/nan/inf. This variant
  has teeth that the random one lacks. On the standalone harness, an upper
  triangle that is not cleared (junk + mirror) turns `tril(LD)` into NaN garbage
  at every n >= 33 (the kernel reads it, e.g. through the row swaps). So it must
  be cleared by assignment, which `tril_out` does: a mask multiply would leave
  `NaN·0 = NaN`.

### 10.3 Missing tests: other

- **Subnormal input** (e.g. f32 entries ~1e-40). Compare with CPU in case
  anything on the GPU path (cuBLAS GEMM?) flushes denormals. **Decided: not
  supported, documented by a NOTE above the L21 update in the kernel.** Findings:
  - No sign of flushing on the GPU. At the subnormal boundary (1e-38 f32, 1e-306
    f64) CUDA matches CPU. Deep in it (1e-40, 1e-43, 1e-310, 1e-320) both give a
    mostly non-finite LD: once a pivot is below 1 / max, `t / d01`, `t / d10`
    and the complex `/= D11` overflow. CPU calls non-rook `?sytrf`/`?hetrf`,
    whose `?lasyf`/`?lahef` form `R1 = ONE / A(K,K)` and `D21 = T / D21` the
    same way (and `?sytrs` forms 1 / D too, so the CPU solve is NaN as well).
  - `c10::complex` division is Smith's algorithm (numpy's variant), but it forms
    `scl = 1 / (c + d * rat)` and multiplies by it, so it overflows for a
    subnormal divisor even when the quotient is representable.
  - LAPACK's fix lives in `?lasyf_rook`/`?lahef_rook` (3.5+): `SFMIN =
    DLAMCH('S')`; 1x1 multiplies by the reciprocal only if `|D| >= SFMIN`, else
    divides element-wise; 2x2 divides inside the loop,
    `T*((D11*W(J,KW-1) - W(J,KW)) / D21)`. The complex `/` there is the
    compiler's: gfortran (`-fcx-fortran-rules`, `expand_complex_div_wide`) does
    Smith's and divides by the denominator.
  - A rook-style fix was prototyped and works: 2x2 loop
    `div(t * (r1 * l0 - l1), d01)`, r0/r1 and the 1x1 column through a Smith
    `ldl::div` that divides by its denominator. All dtypes then finite at
    1e-40/1e-43/1e-310/1e-320, residual (D rescaled by 2^k) within 1.5x of CPU LU
    in the same arithmetic. Not kept; the patch is
    `agent_space/subnormal_div_fix.patch` (`git apply` it, on top of `c15de04fcc3`
    it needs only line offsets).
- **Near-overflow complex magnitude.** cabs1 = |re| + |im| overflows at about
  finfo.max / 2. LAPACK uses the same cabs1, so compare with CPU rather than
  expect success. **Done: `test_cabs1_overflow`** (one entry 0.7 max (1 + i),
  modulus 0.99 max, in column 0 / the middle / the trailing corner; wherever CPU
  gives a finite LD with info 0, so must CUDA). Findings:
  - The normwise residual is useless here: `||A|| ||X||` overflows (residual
    reads 0), and with the overflow avoided it is ~1e-31 / 1e-293 whatever the
    answer, since `||A||` is ~max. LU (`linalg.solve`) fails on these inputs.
  - Componentwise (Oettli-Prager) backward error is ~1/eps on CUDA in nearly all
    cases: the entry lands in a 2x2 pivot as d01, and Smith's denominator
    `c + d (d / c)` overflows to inf, so `r0`, `r1`, `t / d01` and L are 0 instead
    of ~1e-38 / ~1e-308, and the solve loses the components on that block. CPU's
    blocked `?lahef` (MKL, n >= 300) gives the *identical* error; CPU's unblocked
    `?hetf2` (n <= 64, and the trailing columns) is accurate, because it divides
    by the real `|d01|` (`DLAPY2`). Not fixed (as for subnormals), recorded in
    the kernel NOTE above the L21 update. A fix would be `?hetf2`'s: normalize the
    Hermitian 2x2 by `a = |d01|`, `r0 = d00 / a`, `r1 = d11 / a`, `u = d10 / a`,
    `t = 1 / (r0 r1 - 1)`, then `l0' = (t / a)(r1 l0 - u l1)`,
    `l1' = (t / a)(r0 l1 - conj(u) l0)`.
  - zero_diagonal n=64 (complex64 trailing, complex128 mid): CUDA gives a finite
    LD with info 1. With sigma = cabs1 = inf, the pivot test
    `|a_kk| >= alpha lambda (lambda / sigma)` reads `0 >= 0` and takes the zero
    diagonal as a 1x1 pivot. LAPACK has the same formula; CPU is non-finite with
    info 4 there.
  - Whole matrix scaled to 0.25 / 0.9 / 1.2 max: the GEMM updates overflow on
    both sides from n = 64, and which side stays finite at n = 5 or 64 is luck of
    summation order, so not tested.
- **Memory regression test. Done: `test_peak_memory`** (n = 4096, batch () and
  (2,), all dtypes, after a warm-up: peak extra <= 2 copies + 1 MiB). Measured
  2.0001-2.0003 copies (+16.5 KiB). Below ~32 MiB per copy the peak is instead
  1 copy + 32 MiB: upstream `gemm_internal_cublas` takes a fresh 32 MiB workspace
  per GEMM from the caching allocator (`getCurrentCUDABlasHandleWithWorkspace`,
  unless `TORCH_CUBLAS_WORKSPACE_CACHE=1`), while the tril temporary is already
  freed. That is why the test uses n = 4096.
- **Default backend. Done:** dispatch extended and `test_dispatch` (section 5).
  The suite is not run twice, since both routes call the same function.
- **Residual factor. Decided: keep 1.5.** On H100 (CPU solve via MKL) the worst
  real case over `test_cooperative_panel`'s structures at n = 2049/4100/8193 is
  0.47 of the 1.5 bound (complex128 integer_valued n = 4100); integer_valued
  n = 8193 is 0.57x / 0.49x LU. So GB200's 1.33x (section 7) does not reproduce
  here; not isolated whether its CPU solve (ATLAS) or its LU reference differs. A
  30·eps relative perturbation of L clears the 1.5 bound by 1.39-5.1x but the 2.0
  bound by only 1.15-3.9x (weakest: complex128 zero_diagonal n = 8193).
- **Cooperative fallback (optional). Not done:** `cuda.bindings` is not
  installed in the `kernel` env on H100 either.
- **Upstream, separate from this branch:** `check_errors=True` on singular input
  hits an INTERNAL ASSERT (section 5). Already reported (#83624, draft fix
  #197234), nothing posted.
