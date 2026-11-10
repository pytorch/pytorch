# Caller-Supplied Tuning Lists

Extra Triton configurations for Inductor's pointwise, reduction and persistent reduction kernels, benchmarked alongside
the heuristic's own candidates. It applies with or without `max_autotune`, so it also reaches reductions, which at
default settings get a single candidate and are not autotuned at all.

## Configuration

```python
from torch._inductor import config

config.tuning_list.table = {
    "reduction/2": {"configs": [{"XBLOCK": 8, "R0_BLOCK": 128}, {"XBLOCK": 1, "R0_BLOCK": 2048}]},
    "gfx942+pointwise/1": {"configs": [{"XBLOCK": 1024, "num_warps": 8}], "mode": "replace"},
}
```

The table is read when Inductor generates each kernel, and the matching entry is written into the kernel's source, so a
table changed between compiles takes effect on the next one.

## Table Format

Keys are `[<device>+]<category>/<arity>`:

- **device** (optional): `DeviceProperties.cc`, which on ROCm is the architecture name, for example `gfx942`. This is
  not `lookup_table`'s device key, which is the full `gcnArchName` (`gfx942:sramecc+:xnack-`). Backends whose `cc` is a
  numeric compute capability have no device key and are served by device-agnostic entries only.
- **category**: `pointwise`, `reduction` or `persistent_reduction`.
- **arity**: `len(size_hints)`, the number of axes the kernel has.

An entry applies to every kernel with that category and arity, whatever its sizes. Codegen kernels are too many and too
varied for exact-shape keys like `lookup_table`'s, so a list has to serve kernels it was not fitted on, which is why
`append` is the default.

Each value is a dictionary with:

- **configs**: a list of configurations, each naming every block size the key's kernels take, plus optional knobs.
- **mode** (optional):
  - `append` (default): added to the generated candidates, which stay in the benchmark.
  - `replace`: only the supplied candidates are benchmarked; if all of them are dropped for a kernel, the generated
    ones are kept.

### What a configuration names

Block names follow the constructors' positional order, `XBLOCK`, `YBLOCK`, `ZBLOCK`, `R0_BLOCK`, restricted to the axes
the kernel has. A knob left out is derived by the constructor the heuristic uses. A reduction with many loads gets that
constructor's register-pressure limit on every supplied configuration; the heuristic gives it to only some of its own.

| Category | Blocks | Knobs |
| --- | --- | --- |
| `pointwise` | one per axis | `num_warps`, `num_stages`, `num_elements_per_warp`, `waves_per_eu`, `kpack` |
| `reduction` | one per axis | `num_warps` (not with a `y` axis), `num_stages`, `waves_per_eu` |
| `persistent_reduction` | non-reduction axes only (`persistent_reduction/2` names just `XBLOCK`) | `num_warps` (not with a `y` axis), `num_stages`, `waves_per_eu` |

The compiler sets the excluded ones itself: a persistent reduction spans its whole reduction in one block, and a tiled
reduction derives its warp count from the tile. Naming either is an error.

## Lookup Order

The device-specific key is tried first, then the device-agnostic one. Only one entry applies; they do not merge.

## Validation

Keys are checked whenever a kernel is generated, and an entry when a kernel under its key is.

**Fatal**, raising `InvalidTuningList`:

- a key that is not `[<device>+]<category>/<arity>` with one of the three categories
- an entry that is not a dictionary, or carries no `configs`
- an unknown `mode`
- a name that is neither a block size nor a knob for that category, including the exclusions above
- a block size the key's kernels take but the configuration does not name
- a value that is not an integer; NumPy integers are accepted and stored as plain ones
- a block size or `num_warps` that is not a power of 2, which would fail the whole kernel in Triton
- a negative knob

**Dropped with a warning**:

- a configuration the constructors reject for a particular kernel's shape, such as a block above the backend's limit on
  a kernel whose extent is larger still

A block larger than the kernel's extent is clamped to the extent.

## Precedence

Three mechanisms can decide a codegen kernel's configuration:

1. **`config.autotune_lookup_table`** wins outright. It is keyed on a hash of the generated kernel source and applies to
   kernels whose source contains `_fused_`; on a hit nothing is benchmarked.
2. **The `.best_config` autotune cache** holds the winner of a previous run. Its key includes a hash of the candidate
   list, so changing the list misses the stale entry and re-benchmarks.
3. **A supplied tuning list** only adds to or replaces what the heuristics generated.

## Downstream Processing

Supplied candidates go through everything generated ones do:

- filtering for TMA restrictions and for deterministic reductions
- the mix-order reduction rewrite, which overwrites `XBLOCK` and `NUM_STAGES` on every candidate with values derived
  from the split size, so naming a block size has no effect on those kernels
- deduplication: an entry that builds to a configuration already in the list, the heuristic's own included, becomes one

## Not Covered

- Kernels whose reduction block is planned (`strict_reduction_rblock`), since every candidate must agree with the plan.
- Native matmul kernels (`config.triton.native_matmul`), whose configurations come from a matmul-specific constructor.
- Kernels tiled over two reduction axes (`config.triton.tile_reductions`), which share their key with kernels tiled on
  `y` but take different blocks.
- Combo kernels (`config.combo_kernels`) and cooperative reductions.

## Cost

Extra candidates cost benchmarking time on cold builds only; the winner is cached.
