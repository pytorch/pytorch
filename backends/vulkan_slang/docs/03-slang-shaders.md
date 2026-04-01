# Slang Shaders & Compilation Pipeline

## Why Slang as the Sole Shader Language

All GPU shaders in this project are written in **Slang** (https://shader-slang.org/). No GLSL. Reasons:

1. **Autodiff eliminates hand-written backward shaders.** `[Differentiable]` + `bwd_diff()` auto-generates backward-mode derivative propagation at compile time. This cuts backward-pass work by 50-70%.
2. **Generics everywhere.** `<T : IFloat>` covers f32 and f16 for every kernel — including GEMM and attention — without preprocessor hacks or specialization-constant gymnastics.
3. **Full compute shader capabilities.** `groupshared` memory, `GroupMemoryBarrierWithGroupSync()`, workgroup dispatch, subgroup ops — everything needed for tiled GEMM and flash attention. Compiles to the same SPIR-V as GLSL.
4. **Modular compilation.** `import` instead of `#include` pasting. Separate compilation, linking, incremental builds.
5. **Multi-target.** Same source → SPIR-V (Vulkan), CUDA, MSL (Metal), **and C++ (CPU)**. The CPU target is critical for the dev workflow (see `05-cpu-testing.md`).
6. **Production-proven.** Valve ships Counter-Strike 2 and Dota 2 through Slang's SPIR-V codegen. Khronos hosts the project. It's in the Vulkan SDK.
7. **Best-in-class tooling.** VS Code LSP with IntelliSense, RenderDoc debugging, meaningful compile errors at definition site.

For porting GLSL reference code (e.g., llama.cpp's `mul_mm.comp`): Slang provides a GLSL compatibility module and the syntax is close enough that porting is mechanical.

---

## Shader Directory Layout

```
shaders/
├── common/                        # Shared Slang modules
│   ├── types.slang                # Type aliases, IScalarType interface
│   ├── indexing.slang             # N-dim index <-> linear offset (generic)
│   ├── reduce.slang               # Subgroup/workgroup reduce primitives
│   ├── broadcast.slang            # Broadcasting logic for binary ops
│   └── random.slang               # Philox4x32 PRNG module
├── unary/                         # Unary element-wise (all [Differentiable])
│   ├── neg.slang, exp.slang, log.slang, sqrt.slang, rsqrt.slang
│   ├── abs.slang, sign.slang, ceil_floor.slang
│   └── cast.slang                 # dtype conversion (non-differentiable)
├── binary/                        # Binary element-wise with broadcasting
│   ├── add.slang, mul.slang, sub.slang, div.slang, pow.slang
├── activation/                    # Activation functions (all [Differentiable])
│   ├── relu.slang, gelu.slang, silu.slang, sigmoid.slang, tanh.slang
│   └── softmax.slang              # Fused numerically-stable softmax
├── matmul/                        # Matrix multiplication
│   ├── mm_naive.slang             # Baseline (correctness reference)
│   ├── mm_tiled.slang             # Shared-memory tiled GEMM (BM=64,BN=64,BK=16)
│   ├── mm_coopmat.slang           # KHR_cooperative_matrix GEMM
│   └── mm_splitk.slang            # Split-K for tall-skinny matrices
├── conv/                          # Convolution shaders
│   ├── im2col.slang, col2im.slang
│   └── conv2d_direct.slang        # Direct conv for small kernels (3×3, 1×1)
├── attention/                     # Flash-attention style shaders
│   ├── flash_attn_fwd.slang
│   └── flash_attn_bwd.slang       # Recomputation-based backward
├── norm/                          # Normalization (all [Differentiable])
│   ├── layer_norm.slang, batch_norm.slang, group_norm.slang
├── loss/                          # Loss functions ([Differentiable])
│   ├── mse_loss.slang, nll_loss.slang, cross_entropy.slang
├── reduce/                        # Reductions
│   ├── sum.slang, mean.slang, max_min.slang, argmax.slang
├── index/                         # Indexing & scatter/gather
│   ├── gather.slang, scatter.slang, index_select.slang, embedding.slang
├── compare/                       # Comparison ops (non-differentiable)
│   ├── cmp_ops.slang, where.slang
├── pooling/                       # Pooling ops
│   ├── max_pool2d.slang, avg_pool2d.slang, adaptive_avg_pool2d.slang
├── random/                        # RNG shaders (non-differentiable)
│   ├── philox.slang, dropout.slang
└── copy/                          # Memory operations
    ├── copy.slang, fill.slang, contiguous.slang
```

---

## Slang Patterns

### Pattern 1: Element-wise with Autodiff (majority of ops)

```slang
// shaders/activation/gelu.slang
import common.types;

[Differentiable]
float gelu_exact(float x) {
    let k = 0.7978845608f;
    let c = 0.044715f;
    let inner = k * (x + c * x * x * x);
    return 0.5f * x * (1.0f + tanh(inner));
}

[shader("compute")] [numthreads(256, 1, 1)]
void computeMain(
    uniform StructuredBuffer<float> input,
    uniform RWStructuredBuffer<float> output,
    uniform uint numel,
    uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= numel) return;
    output[tid.x] = gelu_exact(input[tid.x]);
}

// Backward entry — compile_shaders.py auto-generates this wrapper
[shader("compute")] [numthreads(256, 1, 1)]
void bwd_computeMain(
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> grad_output,
    uniform RWStructuredBuffer<float> grad_input,
    uniform uint numel,
    uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= numel) return;
    DifferentialPair<float> dp = diffPair(input[tid.x], 0.0f);
    bwd_diff(gelu_exact)(dp, grad_output[tid.x]);
    grad_input[tid.x] = dp.getDifferential();
}
```

### Pattern 2: Tiled GEMM with Groupshared Memory (performance-critical)

```slang
// shaders/matmul/mm_tiled.slang
static const uint BM = 64;
static const uint BN = 64;
static const uint BK = 16;
static const uint TM = 8;
static const uint TN = 8;

groupshared float smem_a[BM * BK];  // Slang groupshared = GLSL shared
groupshared float smem_b[BK * BN];

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uniform StructuredBuffer<float> A,   // [M, K]
    uniform StructuredBuffer<float> B,   // [K, N]
    uniform RWStructuredBuffer<float> C, // [M, N]
    uniform uint M, uniform uint N, uniform uint K,
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID,
    uint tidx : SV_GroupIndex)
{
    uint block_row = gid.x;
    uint block_col = gid.y;

    float reg_c[TM * TN];
    for (uint i = 0; i < TM * TN; i++) reg_c[i] = 0.0f;

    for (uint bk = 0; bk < K; bk += BK)
    {
        // Cooperative load A tile into smem_a
        for (uint i = tidx; i < BM * BK; i += 256) {
            uint row = block_row * BM + i / BK;
            uint col = bk + i % BK;
            smem_a[i] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }
        // Cooperative load B tile into smem_b
        for (uint i = tidx; i < BK * BN; i += 256) {
            uint row = bk + i / BN;
            uint col = block_col * BN + i % BN;
            smem_b[i] = (row < K && col < N) ? B[row * N + col] : 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        uint thread_row = (tidx / (BN / TN)) * TM;
        uint thread_col = (tidx % (BN / TN)) * TN;

        for (uint k = 0; k < BK; k++) {
            for (uint tm = 0; tm < TM; tm++) {
                float a_val = smem_a[(thread_row + tm) * BK + k];
                for (uint tn = 0; tn < TN; tn++) {
                    reg_c[tm * TN + tn] += a_val * smem_b[k * BN + thread_col + tn];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write results
    for (uint tm = 0; tm < TM; tm++) {
        for (uint tn = 0; tn < TN; tn++) {
            uint row = block_row * BM + (tidx / (BN / TN)) * TM + tm;
            uint col = block_col * BN + (tidx % (BN / TN)) * TN + tn;
            if (row < M && col < N)
                C[row * N + col] = reg_c[tm * TN + tn];
        }
    }
}
```

### Pattern 3: Generic Dtype via Slang Generics

```slang
// shaders/binary/add.slang
[Differentiable]
T typed_add<T : IFloat>(T a, T b) { return a + b; }

// f32 variant
[shader("compute")] [numthreads(256, 1, 1)]
void add_f32(uniform StructuredBuffer<float> a, uniform StructuredBuffer<float> b,
             uniform RWStructuredBuffer<float> out, uniform uint numel,
             uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= numel) return;
    out[tid.x] = typed_add<float>(a[tid.x], b[tid.x]);
}

// f16 variant — same logic, different type
[shader("compute")] [numthreads(256, 1, 1)]
void add_f16(uniform StructuredBuffer<half> a, uniform StructuredBuffer<half> b,
             uniform RWStructuredBuffer<half> out, uniform uint numel,
             uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= numel) return;
    out[tid.x] = typed_add<half>(a[tid.x], b[tid.x]);
}
```

---

## Shader Compilation Pipeline

```
compile_shaders.py                     compile_cpu_tests.py
        │                                       │
        ▼                                       ▼
  For each .slang:                        For each .slang:
  ├─ slangc → fwd.spv (forward)          └─ slangc -target cpp → .cpp/.h
  ├─ slangc → bwd.spv (backward)            (CPU-runnable shader math)
  └─ embed as C++ byte arrays
     in csrc/generated/shaders.h         → cpu_tests/generated/
```

### Backward Entry Point Convention

`compile_shaders.py` auto-generates backward wrappers for every `.slang` with `[Differentiable]`. Shader authors only write forward logic. The script produces `{op}_fwd.spv` and `{op}_bwd.spv`.

### Implementation Tasks

- [ ] `shaders/common/types.slang`: `IScalarType` interface, f32/f16 aliases
- [ ] `shaders/common/indexing.slang`: generic N-dim index ↔ flat offset with stride
- [ ] `shaders/common/broadcast.slang`: broadcasting logic as reusable module
- [ ] `shaders/common/reduce.slang`: subgroup/workgroup reduce primitives
- [ ] `shaders/common/random.slang`: Philox4x32 PRNG module
- [ ] `compile_shaders.py`: batch compile, generate forward + backward SPIR-V + C++ test targets
- [ ] `compile_cpu_tests.py`: generates C++ from Slang for CPU-side unit tests

**Testing gate:**
- [ ] `slangc` compiles a trivial compute shader to valid SPIR-V
- [ ] `slangc -target cpp` produces compilable C++ from the same source
- [ ] Backward entry generation works for `[Differentiable]` functions
- [ ] Slang SPIR-V loads and executes correctly on SwiftShader
- [ ] CPU-compiled Slang functions produce same outputs as SPIR-V path
- [ ] Pin Slang version, document in `tools/slang_version.txt`

---

## Slang Autodiff Integration with PyTorch Autograd

PyTorch autograd and Slang autodiff are complementary layers:

- **PyTorch autograd** = op-level graph. Chains gradients across operations.
- **Slang autodiff** = shader-level. Generates the math *inside* each backward kernel.

```
PyTorch autograd graph:
    x → [relu] → [mm] → [softmax] → [cross_entropy] → loss
                                                          │
    loss.backward() walks the graph:                      ▼
    ┌──────────────────────────────────────────────────────┐
    │ cross_entropy_bwd  ← Slang bwd_diff (auto)          │
    │ softmax_bwd        ← Slang bwd_diff (auto)          │
    │ mm_bwd             ← PyTorch autograd (decomposes    │
    │                       to mm + transpose, reuses fwd) │
    │ relu_bwd           ← Slang bwd_diff (auto)          │
    └──────────────────────────────────────────────────────┘
```

### Three Tiers of Backward Support

**Tier 1 — Slang autodiff (majority):** Forward is `[Differentiable]` → backward SPIR-V auto-generated at build time. Covers: all unary, binary, activations, norms, losses, softmax, avg_pool2d, reductions.

**Tier 2 — PyTorch autograd decomposition:** PyTorch decomposes backward into forward ops. E.g., `mm` backward = two more `mm` calls with transposed inputs. No backward shader needed.

**Tier 3 — Hand-written Slang backward:** For complex ops where Slang autodiff can't handle global memory patterns. Use `[BackwardDerivative(custom_fn)]`. Covers: flash attention (recomputation-based), batch_norm training mode, embedding_backward (scatter_add).

---

## Reference Materials

### Slang
- Homepage / GitHub: https://shader-slang.org/ / https://github.com/shader-slang/slang
- Autodiff Guide: https://shader-slang.org/slang/user-guide/autodiff
- Autodiff Tutorial: https://docs.shader-slang.org/en/stable/auto-diff-tutorial-1.html
- Generics: https://shader-slang.org/slang/user-guide/interfaces-generics
- Coming from GLSL: https://shader-slang.org/slang/user-guide/coming-from-glsl
- Cooperative Vectors: https://shader-slang.org/blog/2025/01/30/coop-vec-available/

### GEMM Optimization
- CUDA GEMM Worklog (concepts transfer): https://siboehm.com/articles/22/CUDA-MMM
- Cooperative Matrix: https://developer.nvidia.com/blog/machine-learning-acceleration-vulkan-cooperative-matrices/

### Existing Implementations (Study & Port)
- **llama.cpp ggml-vulkan:** https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-vulkan — Study `mul_mm.comp` GEMM tiling, port to Slang
- **Sascha Willems' Vulkan Samples (Slang):** 170 Slang shaders for Vulkan — excellent reference
