# Custom CUDA LDL kernel — session notes

Handoff notes for `torch.linalg.ldl_factor(x, hermitian=True)` on the
`nikitaved/ldl_factor` branch. Written to be picked up on another machine.

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

The custom kernel is **not** used under the default linalg backend. It lives
under `case LinalgBackend::Cusolver` in `ldl_factor_kernel`
(`aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp`, ~line 575) and is
gated on `hermitian`:

```python
torch.backends.cuda.preferred_linalg_library("cusolver")   # REQUIRED
torch.linalg.ldl_factor_ex(A, hermitian=True)              # hermitian=True REQUIRED
```

With the default backend, `complex && hermitian` goes to **MAGMA** and
everything else to **cuSOLVER**, so the custom kernel never runs.

This cost hours in the first session: a full green test suite was reported as
validating the kernel when it was in fact exercising MAGMA. `test_ldl_custom.py`
forces the backend in `setUp()`.

The gate is on `hermitian` only; **real dtypes reach the kernel too**.
`hermitian=False` (real symmetric or complex symmetric) goes to cuSOLVER, not
to the custom kernel.

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

**Debug scaffolding still in the tree.** `ldl_factor_kernel` has a `// DEBUG`
comment, odd brace/indent formatting, and the hermitian branch wired directly to
the custom kernel with no dtype/size/availability condition. Needs a real
dispatch policy before landing (undecided).

**`check_errors=True` on singular input** raises
`INTERNAL ASSERT FAILED ... Unknown error code: 1` from
`aten/src/ATen/native/BatchLinearAlgebra.cpp:1679`. **Reproduces on CPU** — a
pre-existing gap in the error mapping, unrelated to this kernel. Worth reporting
upstream separately. Not pinned by a test (it is a bug, not a contract).

**Not testable on this machine:** the cooperative cap and the
`cudaErrorCooperativeLaunchTooLarge` fallback in the real library (see section
4; the standalone harness tests the kernel on small grids instead), and GB200
(sm_100), whose panel widths differ (`ldl::panel_width`). The harness covers the
widths themselves (section 7) but not the hardware.

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
- Batch shapes through `(2,2,2)`, non-contiguous batches (strided, permuted),
  batched `lda > n`, `out=`, `nrhs`, non-default stream, input on another GPU,
  CUDA-graph capture (must fail cleanly or reproduce eager), determinism (also
  under concurrent GPU load), half/bfloat16 (`NotImplementedError`).
- NaN/Inf input (column, single entry, all) through the real library.
- Only the lower triangle is read (junk in the upper triangle and in the
  imaginary part of a Hermitian diagonal must not matter), and a Hermitian `D`
  has a real diagonal.
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
- `ldl_solve` on CUDA for real dtypes with the custom kernel's output (both
  `hermitian` flags).

### Accuracy metric
LAPACK's scaled solve residual, as in `xPOT02`/`xSYT02` (from memory, not checked
against the source): `||B - AX||_1 / (||A||_1 ||X||_1 eps)` with `eps =
finfo.eps / 2` (LAPACK's `dlamch('E')` counts rounding). LAPACK's own drivers use a
threshold of 30. Measured on all structures up to n=2049 it is 0.1 to 22 for CUDA
and CPU LAPACK alike. It is **not** independent of `n`: on the hardest inputs it
grows like `0.5·sqrt(n)` on both backends (22 at 2049, 32 at 4100, 42 at 8193, 98
at 40000), the usual random-walk accumulation of rounding errors. So the bound is
`max(30, 0.7·sqrt(n))` (`_resid_threshold`), about 1.4x to 1.5x above the worst
measured case at every size. The earlier tolerances (`1000·n·eps` with
max-norms, and `max(50·cpu, 200·n·eps)`) were ad hoc and 3 orders of magnitude
loose; they are gone.

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
