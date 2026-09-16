# Custom CUDA LDL kernel — session notes

Handoff notes for `torch.linalg.ldl_factor(x, hermitian=True)` on the
`nikitaved/ldl_factor` branch. Written to be picked up on another machine.

Branch state at time of writing: HEAD = `2929645c72c`, working tree clean.

```
2929645c72c make almost all test pass
8b4e88e71c5 tests plus fixes -- needs review
be25a6d9610 add tmp test
459a7097d88 bulletproof sycs
36fcf5eb194 minor
42e254f7b55 drop D symmetry enforcement
```

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

This cost hours in the session: a full green test suite was reported as
validating the kernel when it was in fact exercising MAGMA. Forcing the backend
turned 3792 passing tests into 1221 failures. `test_ldl_custom.py` now forces it
in `setUp()`.

Quick discriminator — `n=1` is broken in the custom kernel only:

| backend | `n=1, hermitian=True` |
|---|---|
| Default | `piv=[1]` (fine) |
| Cusolver | `piv=[0]` ← custom kernel |

The gate is on `hermitian` only; **real dtypes reach the kernel too**.

---

## 2. Environment & build

```bash
eval "$(micromamba shell hook --shell bash)" && micromamba activate kernel
cd /home/nik/pytorch
python setup.py develop          # what was used in this session
```

Note `CLAUDE.md` says to build only via `pip install -e . -v --no-build-isolation`;
`setup.py develop` was used at the user's explicit instruction.

- Incremental rebuild of the kernel file is ~5 ninja steps, under a minute.
- The kernel lives in **`libtorch_cuda_linalg.so`**, not `libtorch_cuda.so`.
  Check that one's timestamp when verifying a rebuild landed.
- If a rebuild appears to do nothing, `touch` the `.cu` — a stale-object state
  bit once (object newer than source, so ninja skipped it, while the loaded
  library predated both).
- Hardware used: 4× H100 (sm_90). `build/CMakeCache.txt` shows
  `CMAKE_CUDA_ARCHITECTURES=75`, which is a **red herring** — PyTorch sets archs
  via its own `TORCH_CUDA_ARCH_LIST` path. `torch.cuda.get_arch_list()` returns
  `['sm_90']`, which is the real answer.

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
moves values but not the debt.

### Validation (on the real kernel, `cusolver` forced)
Backward error `‖AX−B‖/(‖A‖‖X‖+‖B‖)` vs CPU LAPACK, well-conditioned indefinite:

| n | float64 | complex128 | float32 | complex64 |
|---|---|---|---|---|
| 33 | 2.2e-15 | 2.3e-15 | 7.8e-07 | 2.0e-06 |
| 257 | 1.9e-13 | 3.6e-13 | 3.9e-05 | 1.7e-04 |
| 1023 | 2.2e-12 | 1.2e-11 | 1.0e-03 | 4.5e-03 |

---

## 4. Other fixes made this session

**Dropped the forced-real `D`** (`42e254f7b55`). The kernel used to force
`Im(D11)=0`, `Im(D22)=0` and take `real()` of `D11`/`det`. In the in-place scheme
`U12` is an independently stored upper triangle, so realifying `D` desynced it
from the `U12` the trailing update consumes. Cost ~6 orders of magnitude:
`1.96e-06` → `2.18e-12` backward error at n=512 in the prototype. Note LAPACK
*does* realify — it's wrong **here** specifically, so don't "fix" it back.

**Barriers** (`459a7097d88`). Three `__syncthreads()` added: before the `sigma`
search (shared-memory reuse between the two `find_pivot_row` calls), after the
`if (tid == 0)` pivot/info block (its reads race the swap writes), and at the end
of the loop body (trailing writes vs the next iteration's pivot read — the one
with a genuinely wide window).

**`info` for 2×2 pivots.** The old test `dLD[piv,piv] == 0` is the *normal* state
for a 2×2 block — a zero diagonal is why Bunch-Kaufman chose one. Now split by
rank where `D` is known: `|D11| == 0` for 1×1, `|det(D)| == 0` for 2×2. This was
~1086 of the 1221 failures.

**Illegal memory access.** `AGGREGATE_ARGMAX` tests `other_val > val`, false for
NaN, so an all-NaN column leaves `my_idx = -1`; `LinOff(t, -1, lda)` then reads
off the front of the buffer. Guarded with `ilambda < 0` falling back to a 1×1
pivot. Reproduced as a hard `CUDA error: an illegal memory access`.

**Batching.** The kernel has `grid = dim3(1,1,1)` and a single `dLD` pointer, so
it silently factored only batch element 0 and returned uninitialized pivots for
the rest. Now looped in `BatchLinearAlgebra.cpp` (per the user's preference —
kept out of the kernel file):

```cpp
const auto batch_size = batchCount(LD);
if (batch_size == 1) { ldl_factor_blas3_kernel(LD, pivots, info, hermitian); }
else {
  auto LD_3d = LD.view({-1, LD.size(-2), LD.size(-1)});
  ...
  for (const auto i : c10::irange(batch_size))
    ldl_factor_blas3_kernel(LD_3d[i], pivots_2d[i], info_1d[i], hermitian);
}
```

`view` not `reshape`: `reshape` would silently copy on an awkward layout and the
in-place result would be discarded; `view` throws instead.

---

## 5. Open issues

**`n = 1`.** `while (step < n - 1)` never enters, so `dipiv[0]` is never written
and comes back as uninitialized memory. Known; tests start at `n = 2`.

**`info` missed when an exactly-zero column crosses a panel boundary.**
Fully diagnosed, not fixed. The undo's `+= corr` adds the *departing* column's
correction — magnitude O(1), measured 0.68 to 6.8 — to a column whose correct
value is exactly `0`, relying on the panel-end GEMM to subtract the same
quantity. cuBLAS computes it in a different order, leaving `≈ eps·|corr|`
(~1e-17) where `0.0` belongs, so the zero pivot is never exact and `info` stays 0.

The rule is exact: zero column stays within its panel → `info` correct every
time; crosses a panel boundary with an update pending → missed every time.

This is a three-way bind, tested and confirmed in `agent_space/design4.py`:

| approach | ‖LD−LAPACK‖ | pivots match | zero col exact |
|---|---|---|---|
| additive (current) | 1.1e-12 | yes | no |
| swap `U12` rows | 8.7e+03 | no | yes |
| swap `U12` + `L` rows | 5.6e+01 | no | yes |

The deferred GEMM needs operands carrying the later permutation; the output `L`
needs those same lower-triangle entries left unpermuted; exact preservation
needs undo and GEMM to cancel bit-for-bit. Any two, not all three.

Options discussed, none implemented:
- **(a)** detect at undo time (the landing column is verifiably, exactly zero
  there) and set `info = ilambda + 1`. ~10 lines, near-free — the undo already
  streams that column; needs the check over the *full* active column
  `[curr_step, n)`, not just `[panel_end, n)`, or it false-positives. Downside:
  the index isn't LAPACK's (which names the eventual pivot position).
  A diagonal-only check is **not** viable — `zero_diagonal` matrices are
  nonsingular with every diagonal entry zero.
- **(b)** flag-and-defer — matches LAPACK's index but needs an `n`-sized array
  swapped on every interchange, and `LD` still diverges. Judged not worth it.
- **(c)** flag-and-restore — re-zero the column after the GEMM. Fully correct,
  same array cost.
- **(d)** document and move on.

**Accuracy gap vs CPU grows with `n`.** CUDA/CPU backward-error ratio runs 3.6 →
76 (float64) and 1.9 → 180 (complex128) from n=33 to 1023. Not a defect — it
matches the Python prototype's prediction exactly (predicted ~50 at n=512,
measured 47.5), and comes from the two triangles drifting apart in the in-place
scheme. The `W` scheme closes it.

**`check_errors=True` on singular input** raises
`INTERNAL ASSERT FAILED ... Unknown error code: 1` from
`aten/src/ATen/native/BatchLinearAlgebra.cpp:1679`. **Reproduces on CPU** — a
pre-existing gap in the error mapping, unrelated to this kernel. Worth reporting
upstream separately.

**Debug scaffolding still in the tree.** `ldl_factor_kernel` has a `// DEBUG`
comment and the hermitian branch wired directly to the custom kernel. Needs
proper dispatch before landing.

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
the staleness bug, the undo/GEMM cancellation problem, and the accuracy gap.
Cost: a rewrite of `ldl_diagonal_panel_fused_kernel` plus two `n × nb`
workspaces (~10 MB at n=10000, nb=32, complex128).

One trap found the hard way: in the `W` formulation the 2×2 `D` **must** be
symmetrized (`D[0,1] = conj(D[1,0])`). Without it, n=512 backward error is
`3.09e-07`; with it, `3.45e-14`. This is the opposite of the in-place scheme,
where forcing `D` real is harmful. The two schemes have opposite requirements
because `W`-based `U` is derived by conjugation while in-place `U12` is stored
independently.

---

## 7. Tests

`test_ldl_custom.py` at repo root — **tracked** (force-added; `.gitignore:409`
has `/test_*.py`, so re-adding needs `git add -f`).

```bash
eval "$(micromamba shell hook --shell bash)" && micromamba activate kernel
python test_ldl_custom.py            # ~5100 tests, ~95 s
python test_ldl_custom.py -k determinism
```

Forces the cusolver backend in `setUp`, restores in `tearDown`.

Covers: 17 matrix structures; shapes 2→4097 (dense around 32/64/512 boundaries);
all 230 compositions of `n` into 1s and 2s for n=2..10 with the achieved pivot
layout **asserted**; ~90% 2×2 configurations at n=150..513; batch shapes through
`(2,2,2)` including an element-wise independence check; determinism (12 bitwise
repeats, plus 10 under concurrent GPU load); `lda > n` views; empty; `out=`;
`nrhs`; `info` semantics.

**Two rules the suite encodes, both learned painfully:**

1. **Never validate by comparing pivots or `LD` against CPU.** Bunch-Kaufman
   decisions sit on thresholds, so a different summation order legitimately
   picks a different, equally valid pivot sequence — after which `L` and `D`
   differ wholesale. From n≈260 up, CUDA and CPU routinely disagree on pivots
   while both factorizations are fine. Validate with the solve residual. An
   entire round of "failures" in this session was this mistake.
2. **The solve must run on CPU.** `torch.linalg.ldl_solve` rejects complex with
   `hermitian=True` on CUDA — exactly the combination the kernel exists for.

Controls that matter: `positive_definite` and `diagonal_only` trigger **zero**
interchanges, so a broken interchange path cannot hide behind them. Verified
coverage at n=257: `indefinite` 155 out-of-panel pivots, `zero_diagonal` 165,
`out_of_panel_heavy` 149, PD 0.

Weak spots, deliberate: `arrow` yields only 1–2 interchanges (its diagonal
dominates); `tridiagonal` structurally cannot produce an out-of-panel pivot.

`test.py` at root is the user's original scratch repro (n ∈ {37,111,247,1023}).
Note it does **not** set the cusolver backend, so by default it exercises MAGMA.

---

## 8. Python prototypes — `agent_space/` (gitignored, `.gitignore:306`)

These will **not** transfer with a git clone. Copy them separately if needed.

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
- **Pivot equality is not a correctness criterion.** Use backward error.
- **Prototype in Python against LAPACK before writing CUDA.** Every design that
  was validated in `agent_space/` first worked; the one written straight into
  CUDA (design 2) had bugs that took a rebuild cycle each to find.
- **A passing test proves nothing until you verify it exercises the path.**
  Instrumenting for out-of-panel-pivot counts per structure is what gave
  confidence the suite was real.
- **Verify the build actually relinked** before interpreting results. Garbage
  pivots were once a stale library, not a kernel bug.
