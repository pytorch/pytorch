# ROCm backward implementation

`flex_rocm_backward.py` owns gfx950 profile selection, eligibility checks, and separate backward phase configuration.
`templates/flex_backwards.py.jinja` contains the common backward body.
Its extension points use the common implementation unless an override is defined.
`templates/flex_backwards_rocm.py.jinja` defines ROCm overrides and additional helpers.
The ROCm template combines these overrides with the common body.
Updates to shared code therefore reach both templates.
Changed operations still require review of the corresponding ROCm overrides.

Unoptimized candidates use the generic template and grid, including gfx950 opt-out and unsupported inputs.
Explicit fused optimizations use the ROCm template only while their validated options remain enabled.
Qualified split paths use the ROCm template for their separate phases.
Both templates use `templates/utilities.py.jinja` for checked loads and sparse-block offsets.

The controls below are internal implementation details for tests and tuning.
Use `BACKEND` and `bwd_ROCM_SEPARATE_DQ=False` through `FlexKernelOptions` for backend selection and automatic-profile opt-out.

With `max_autotune`, legal BF16 D128 and D256 split plans compete with the fused candidates.
Candidate eligibility permits different positive batch sizes and sequence lengths divisible by 128.
D128 supports GQA8 with a default dense mask or a simple causal mask.
D256 supports GQA1 or GQA4 with the default dense mask.
The existing layout, dtype, mask, and option guards still apply.
The measured default profile table does not restrict these candidates.
Existing measured profiles also provide split candidates within their validated input limits.
These candidates retain their measured phase configurations.
Plans that require partial-gradient workspace and reductions remain excluded from this candidate pool.

Each split candidate benchmarks dK/dV and dQ in one launch sequence.
The tuner compares that sequence with the retained fused candidates.
Every candidate shares the same Delta computation outside this comparison.
These candidates require no partial-gradient workspace or reduction.
Subprocess benchmarks reconstruct the same sparse metadata as in-process benchmarks.
Simple causal candidates use diagonal partial blocks and triangular full blocks during benchmarking.
Explicit backward controls and custom backward choice pools retain their existing behavior.
The measured default profiles still apply outside `max_autotune`.

Automatic FP16 D64 selection uses four owners per KV head with FP32 dK/dV partials.
The reducer adds these partials before dK scaling and the final FP16 conversion.
Explicit `bwd_ROCM_GQA_SPLIT_DKDV=True` requests retain two owners per KV head.

Automatic D96 profiles with four query heads per KV head use generic phase1 dots.
At length 2048, phase1 uses tiles `(32, 32, 128, 32)`, four warps, and one stage.
At length 4096, phase1 uses tiles `(64, 64, 128, 32)`, four warps, and one stage.
Their separate dQ phases keep tiles `(32, 128, 128, 32)`, four warps, and two stages.
The D96 profiles with one query head per KV head retain panel dots and their existing dQ controls.
Explicit D96 panel requests retain the panel path described below.

The automatic BF16 D192 profile `(4, 10, 10, 4096)` uses `matrix_instr_nonkdim=16` for both backward phases.
Other automatic D192 profiles retain `matrix_instr_nonkdim=0`.
Its tiles remain `(16, 64, 128, 32)`, with four warps and two stages.

The automatic D256 profile with four query heads per KV head uses phase1 tiles `(32, 64, 128, 32)`.
Its separate dQ phase keeps tiles `(16, 64, 128, 32)`.
Both phases use four warps and two stages.
The M32 phase1 kernel keeps 128-column QK and dP panels.
The M32 phase1 kernel computes dS from FP32 probabilities before it converts P and dS to BF16.
It then accumulates dV and dK.
`XCD_REMAP` changes the phase1 batch/head program order.
Explicit D256 requests retain the tile requirements described below.

The `bwd_ROCM_*` options and `bwd_XCD_REMAP` below are internal experimental controls.
They support testing and tuning, but they are not part of the public `FlexKernelOptions` type.
Use `BACKEND` and `bwd_ROCM_SEPARATE_DQ` for public control.
Use `BACKEND="TRITON"` to select this path explicitly.
These controls have no backward compatibility guarantee.
The compiler checks each request before selecting a specialized kernel.
An unsupported request uses the applicable existing kernel.
Unsupported combinations can retain independent options that remain valid.
Invalid tile dimensions still raise the usual kernel-options error.

The specialized split kernels require an identity `score_mod`, no captured tensors, and contiguous tensors in `(batch, heads, length, dimension)` order.
The D96, D192, and D256 dense split kernels require the default dense `BlockMask`.
The D128 coverage and traversal profiles require a causal mask equivalent to `q_idx >= kv_idx`.
They require equal query and KV lengths and 128-by-128 blocks.
Their activation also requires no gradient for the auxiliary log-sum-exp output.
The general BF16 D128 split path supports other masks and sequence lengths.
Unknown shapes or layouts do not activate a specialized kernel.

In the tables below, tiles mean `(BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2)`.
`Hq/Hkv` specifies the number of query heads and KV heads.

| Specialized path | Dtype | Head dimension | Supported `(batch, Hq, Hkv, length)` | Tiles | Warps / stages |
| --- | --- | --- | --- | --- | --- |
| D96 panels | BF16 | 96 | `(4, 21, 21, 2048/4096)`, `(2, 8, 2, 2048/4096)` | `(32, 128, 128, 32)` | 4 / 2 |
| D192 panels | BF16 | 192 | `(4, 10, 10, 2048/4096)`, `(2, 8, 2, 2048)` | `(16, 64, 128, 32)` | 4 / 2 |
| D256 dense split | BF16 | 256 | Positive batch and lengths; `Hq/Hkv` ratio 1 or 4 | `(16, 64, 64, 16)` or `(16, 64, 128, 32)` | 4 / 2 |
| D128 split | BF16 | 128 | Positive batch and lengths; `Hq/Hkv` ratio 8 | `(32, 128, 128, 32)` | 4 / 2 |
| D128 FP16 coverage | FP16 | 128 | `(4, 64, 8, 2048/4096)`, `(2, 64, 8, 2048/4096)`, `(1, 64, 8, 4096)` | `(32, 128, 128, 32)` | 4 / 2 |
| D128 FP32 coverage | FP32 | 128 | `(1/2/4, 64, 8, 2048/4096)` | `(16, 64, 64, 16)` | 4 / 1 |

Explicit split selection requires one backward configuration.
The split autotune candidates above also support multiple fused configurations.
The backward choices hook uses the fused kernel interface.
When the hook changes its candidate pool, fused choices take precedence over split execution.
Explicit D96 and D192 panel requests require `matrix_instr_nonkdim=0`, `waves_per_eu=0`, and `kpack=1`.
The D128 coverage and traversal profiles require `matrix_instr_nonkdim=0` and `waves_per_eu=0`.
The BF16 traversal profile and FP16 coverage profile support `kpack` values 1 or 2.
FP32 coverage requires `kpack=1`.
The FP32 profile requires IEEE dot-product precision.
ROCm uses IEEE precision by default.

The D256 split profile requires both `bwd_ROCM_SEPARATE_DQ` and `bwd_ROCM_D256_PANEL_DOTS`.
The BF16 D128 split profile requires `bwd_ROCM_SEPARATE_DQ`.
Its measured performance improvements also use early dP and XCD remapping.
The D128 coverage options need three additional requests: `bwd_ROCM_SEPARATE_DQ`, `bwd_ROCM_EARLY_DP`, and `bwd_XCD_REMAP`.
They also require `bwd_USE_TMA=False` and `bwd_PRESCALE_QK=False`.
`bwd_BLOCKS_ARE_CONTIGUOUS` must be a Boolean; `False` retains indexed traversal.
The FP16 and FP32 coverage options cannot activate together.
Panel, snake, GQA-split, probability-compaction, D96 dQ, and D192 dQ requests disable D128 coverage.
Custom consumer groups, `waves_per_simd`, and `llvm_fn_attrs` also disable D128 coverage.

Set `bwd_ROCM_D128_REVERSE_DQ_ROWS=True` to request reverse dQ traversal with an active D128 profile.
Its BF16 scope is limited to `(batch, Hq, Hkv, length)` values `(4, 64, 8, 1024/2048/4096)` and `(2, 64, 8, 4096)`.
Set `bwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64=True` with reverse dQ traversal to request smaller dV accumulator views.
Query-tile-first FP16 kernels use 32-column fragments.
Other qualified kernels use two 64-column views.
These profiles accept finite numeric scales.
The compiler can select query-tile-first dK/dV traversal for all shapes in the FP16 D128 row.
Its FP32 scope includes `(1/2/4, 64, 8, 4096)` and `(4, 64, 8, 2048)`.
This selection supports indexed and contiguous block traversal.
It requires `kpack=1` and contiguous `BlockMask` metadata broadcast across batch and heads, with full blocks.
Its qualified scales are exactly `0.08838834764831843`, `0.0`, and `-0.125`.
The first value is the default D128 scale, `1 / math.sqrt(128)`.
Other scales or metadata use the applicable dV-view kernel.
The compiler controls query-tile-first traversal; it has no user option.

For example, the following configuration requests FP16 D128 causal GQA coverage and both optional traversal changes.
The inputs must satisfy a shape in the FP16 row above.
The causal `BlockMask` must use `BLOCK_SIZE=128`.

```python
kernel_options = {
    "BACKEND": "TRITON",
    "bwd_ROCM_D128_FP16_COVERAGE": True,
    "bwd_ROCM_SEPARATE_DQ": True,
    "bwd_ROCM_EARLY_DP": True,
    "bwd_XCD_REMAP": True,
    "bwd_ROCM_D128_REVERSE_DQ_ROWS": True,
    "bwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64": True,
    "bwd_BLOCKS_ARE_CONTIGUOUS": True,
    "bwd_USE_TMA": False,
    "bwd_PRESCALE_QK": False,
    "bwd_BLOCK_M1": 32,
    "bwd_BLOCK_N1": 128,
    "bwd_BLOCK_M2": 128,
    "bwd_BLOCK_N2": 32,
    "bwd_num_warps": 4,
    "bwd_num_stages": 2,
}
```

For FP32, replace the coverage option with `bwd_ROCM_D128_FP32_BALANCED64_COVERAGE=True`.
Use tiles `(16, 64, 64, 16)` and one stage.

Other combinations have separate activation requirements:

| Request | Additional requirements |
| --- | --- |
| `bwd_ROCM_D192_PANEL_DOTS` | D192 dense profile; `USE_TMA`, `PRESCALE_QK`, general early dP, XCD remapping, and other panel requests disabled |
| `bwd_ROCM_D192_DQ_EARLY_DP` | Active D192 panel profile |
| `bwd_ROCM_D96_PANEL_DOTS` | D96 dense profile; the same exclusions as D192 panels |
| `bwd_ROCM_D96_DQ_EARLY_DP` | Active D96 panel profile; batch 4, 21 query and KV heads; contiguous blocks |
| `bwd_ROCM_D96_DQ_UNROLL2` | Active D96 early-dP dQ profile |
| `bwd_ROCM_EARLY_PROBABILITY_COMPACT` | Active D256 dense split; panel dots and early dP; tiles `(16, 64, 128, 32)` |
| `bwd_ROCM_GQA_SPLIT_DKDV` | FP16 D64 dense attention; four query heads per KV head; tiles `(32, 128, 128, 32)`, four warps, two stages |
| `bwd_ROCM_XCD_SNAKE` | BF16 D256 causal attention; one or four query heads per KV head; tiles `(32, 64, 64, 64)`, four warps, one stage |

The GQA-split and snake paths also require contiguous inputs, an identity score modification, no captures, and one backward configuration.
They exclude tensor descriptors, QK prescaling, and custom consumer groups.
Snake traversal excludes panel dots, separate dQ, GQA splitting, early dP, and XCD remapping.
The GQA-split path reduces FP32 partial dK/dV gradients after its separate kernels.
