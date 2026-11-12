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

Keys are `[<device>+]<category>/<num_axes>`:

- **device** (optional): `DeviceProperties.cc`, which is the architecture name on ROCm, for example `gfx942`, and the
  compute capability as an integer on CUDA, for example `90` for sm_90. The ROCm name is not `lookup_table`'s device
  key, which is the full `gcnArchName` (`gfx942:sramecc+:xnack-`). Other backends have no device key and are served by
  device-agnostic entries only.
- **category**: `pointwise`, `reduction` or `persistent_reduction`.
- **num_axes**: `len(size_hints)`, the number of axes the kernel iterates over, reduction axes included, so
  `reduction/2` is `x` and `r0_`. Read it off the generated kernel's `size_hints` rather than the tensor's shape, since
  Inductor often flattens dimensions into one axis.

An entry applies to every kernel with that category and number of axes, whatever its sizes. Codegen kernels are too
many and too varied for exact-shape keys like `lookup_table`'s, so a list has to serve kernels it was not fitted on,
which is why `append` is the default.

Each value is a dictionary with:

- **configs**: a list of configurations, each naming every block size the key's kernels take, plus optional knobs.
- **mode** (optional):
  - `append` (default): added to the generated candidates, which stay in the benchmark.
  - `replace`: only the supplied candidates are benchmarked; if all of them are dropped for a kernel, the generated
    ones are kept.

### What a configuration names

The block names are `XBLOCK`, `YBLOCK`, `ZBLOCK` and `R0_BLOCK`, for the axes the kernel has. A knob left out is
derived by the constructor the heuristic uses. A reduction with many loads gets that constructor's register-pressure
limit on every supplied configuration; the heuristic gives it to only some of its own.

| Category | Blocks | Knobs |
| --- | --- | --- |
| `pointwise` | one per axis | `num_warps`, `num_stages`, `num_elements_per_warp`, `waves_per_eu`, `kpack` |
| `reduction` | one per axis | `num_warps` (not with a `y` axis), `num_stages`, `waves_per_eu` |
| `persistent_reduction` | non-reduction axes only (`persistent_reduction/2` names just `XBLOCK`) | `num_warps` (not with a `y` axis), `num_stages`, `waves_per_eu` |

The compiler sets the excluded ones itself: a persistent reduction spans its whole reduction in one block, and a tiled
reduction derives its warp count from the tile. Naming either is an error.

## Lookup Order

The device-specific key is tried first, then the device-agnostic one. Only one entry applies; they do not merge.

## Choices Handler

The table is the default `InductorChoices.triton_kernel_configs`, which codegen calls as it generates each pointwise and
reduction kernel, with its heuristic name (`pointwise`, `reduction`, `persistent_reduction` and others), `size_hints`,
`triton_meta` and `inductor_meta`, and which returns an entry or `None`. Overriding it keys entries on anything those
carry, such as size buckets or the reduction hint:

```python
from torch._inductor.choices import InductorChoices, register_inductor_choices


class MyChoices(InductorChoices):
    def triton_kernel_configs(self, heuristic, size_hints, triton_meta, inductor_meta):
        if heuristic == "reduction" and len(size_hints) == 2 and size_hints["r0_"] >= 8192:
            return {"configs": [{"XBLOCK": 1, "R0_BLOCK": 4096}]}
        return super().triton_kernel_configs(heuristic, size_hints, triton_meta, inductor_meta)

    def uuid(self):
        return ("my_choices", 1)


register_inductor_choices("my_choices", MyChoices)
```

An entry from a handler is validated as a table entry is, and does not apply to the kernels under Not Covered. `uuid()`
puts the handler in the FX graph cache key, so change it when the entries change; a handler installed with
`V.set_choices_handler` alone is not in the key.

## Validation

Keys are checked whenever a kernel is generated, and an entry when a kernel under its key is.

**Fatal**, raising `InvalidTuningList`:

- a key that is not `[<device>+]<category>/<num_axes>` with one of the three categories
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

Supplied candidates are built by the device's heuristic class, with the constructor it uses for its own, and go through
everything generated ones do:

- in `append` mode, the heuristic's own post-processing of its list, such as ROCm's `_finalize_configs`; reductions
  skip it where the heuristic returns early for an `INNER`, `OUTER` or `OUTER_TINY` hint
- filtering for TMA restrictions and for deterministic reductions
- the mix-order reduction rewrite, which overwrites `XBLOCK` and `NUM_STAGES` on every candidate with values derived
  from the split size, so naming a block size has no effect on those kernels
- deduplication: an entry that builds to a configuration already in the list, the heuristic's own included, becomes one

## Not Covered

- Kernels whose reduction block is planned (`strict_reduction_rblock`), since every candidate must agree with the plan.
- Native matmul kernels (`config.triton.native_matmul`), whose configurations come from a matmul-specific constructor.
- Kernels tiled over two reduction axes (`config.triton.tile_reductions`), which share their key with kernels tiled on
  `y` but take different blocks.
- Cooperative reductions, which are looked up as the reduction or persistent reduction they would otherwise be.
- Split scans, fixed-config kernels and combo kernels (`config.combo_kernels`).

An entry that matches a planned-block, native matmul, two-reduction-axis or cooperative kernel is skipped with a debug
message on the `torch._inductor.tuning_list.table` logger saying why, which `TORCH_LOGS="+inductor"` shows.

## Cost

Supplied candidates cost compile and benchmark time on cold builds. At default settings most reductions, `INNER` ones in
particular, get a single candidate that is compiled and launched without benchmarking; an `append` entry turns each
kernel under its key into one that compiles every candidate and benchmarks them on its first launch. The winner is
cached, keyed on the candidate list, so later builds compile only the winner.
