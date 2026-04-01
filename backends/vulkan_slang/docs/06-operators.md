# Operator Implementations

All operators are implemented as Slang shaders dispatched via Vulkan compute pipelines. This document covers forward-pass ops (Stage 2) and backward-pass ops (Stage 3).

---

## Stage 2: Core Forward Operations

**Goal:** Run inference on simple models (MLP, small CNN). All correctness verified on SwiftShader.

### 2.1 Element-wise Operations (Slang — all `[Differentiable]`)

**Unary ops:**
- [ ] `neg`, `abs`, `exp`, `log`, `sqrt`, `rsqrt`
- [ ] `ceil`, `floor`, `round`, `sign`
- [ ] CPU shader tests for each (Tier 1)
- [ ] SwiftShader integration tests for each (Tier 2)

**Binary ops (with broadcasting):**
- [ ] `add`, `sub`, `mul`, `div`, `pow`, `fmod`, `remainder`
- [ ] CPU shader tests for each (Tier 1)
- [ ] SwiftShader integration tests for each (Tier 2)

**Comparison ops (non-differentiable):**
- [ ] `eq`, `ne`, `lt`, `gt`, `le`, `ge`, `where`

**In-place variants:**
- [ ] `add_`, `mul_`, `sub_`, `div_`

**Testing gate:**
- [ ] Every unary op: `torch.testing.assert_close(op(x.vulkan()).cpu(), op(x), rtol=1e-5, atol=1e-5)`
- [ ] Every binary op: broadcasting with mismatched shapes (e.g., `[64,64] + [64,1]`)
- [ ] Edge cases: NaN propagation, inf handling, zero division behavior

### 2.2 Copy & Memory Operations

- [ ] `copy_`: Vulkan→Vulkan, CPU→Vulkan, Vulkan→CPU
- [ ] `clone`, `contiguous`
- [ ] `to(dtype)` — dtype conversion
- [ ] `fill_`, `zero_`
- [ ] `empty_strided`

**Testing gate:**
- [ ] CPU→Vulkan→CPU round-trip preserves values exactly (f32)
- [ ] `clone()` produces independent copy (modifying clone doesn't affect original)
- [ ] `contiguous()` on non-contiguous tensor produces correct layout

### 2.3 Shape Operations (Zero-Copy Where Possible)

- [ ] `view`/`reshape` — metadata-only when contiguous
- [ ] `permute`/`transpose` — stride manipulation
- [ ] `expand` — zero-copy broadcast
- [ ] `cat`/`stack` — requires copy
- [ ] `split`/`chunk` — view-based
- [ ] `unsqueeze`/`squeeze`/`flatten`
- [ ] `narrow`/`slice`

**Testing gate:**
- [ ] `view` on contiguous tensor is zero-copy (same data_ptr)
- [ ] `permute` + `contiguous()` produces correct physical layout
- [ ] `cat` along each dimension produces correct result
- [ ] `view` on non-contiguous tensor correctly triggers copy or error

### 2.4 Reduction Operations (Slang)

- [ ] `sum`, `mean`, `prod` (along dim, keepdim)
- [ ] `max`, `min` (values + indices)
- [ ] `argmax`, `argmin`
- [ ] `any`, `all`, `norm`

**Testing gate:**
- [ ] Reduction along each dimension matches CPU result
- [ ] `keepdim=True` preserves correct shape
- [ ] Full reduction (no dim) works
- [ ] Empty tensor edge case handled

### 2.5 BLAS Operations — Slang with Groupshared Tiling

**`mm` (matrix multiply):**
- [ ] `mm_naive.slang` — simple, no tiling (correctness reference)
- [ ] `mm_tiled.slang` — BM=64, BN=64, BK=16 shared-memory tiling
- [ ] f32 and f16 variants via Slang generics
- [ ] CPU test: compile `mm_naive.slang` to C++, verify against reference
- [ ] SwiftShader test: verify tiled version matches naive version

**`mm_coopmat.slang`:**
- [ ] KHR_cooperative_matrix (optional, skip on SwiftShader)

**Other BLAS:**
- [ ] `bmm` — batched matrix multiply
- [ ] `addmm` — fused add + matmul
- [ ] `linear` — addmm wrapper

**Testing gate:**
- [ ] 64×64 matmul matches `torch.mm` on CPU within rtol=1e-4
- [ ] Non-square sizes: [128,64] × [64,256]
- [ ] Tall-skinny: [4096,16] × [16,4096]
- [ ] Batch matmul: [8,64,64] × [8,64,64]
- [ ] f16 variant produces reasonable results (higher tolerance)

### 2.6 Activation Functions (Slang — all `[Differentiable]`)

- [ ] `relu`, `leaky_relu`, `elu`, `selu`, `prelu`
- [ ] `sigmoid`, `tanh`
- [ ] `gelu`, `silu`/`swish`
- [ ] `softmax`, `log_softmax` (fused numerically-stable)

**Testing gate:**
- [ ] Each activation: forward matches CPU reference
- [ ] Softmax: numerically stable (large values don't overflow)
- [ ] Softmax: rows sum to 1.0

### 2.7 Pooling (Slang)

- [ ] `max_pool2d` (with indices for backward)
- [ ] `avg_pool2d` (`[Differentiable]`)
- [ ] `adaptive_avg_pool2d`

**Testing gate:**
- [ ] Pool output shape correct for various kernel/stride/padding combos
- [ ] `max_pool2d` indices correct (verified against CPU)

### 2.8 Convolution (Slang)

- [ ] `im2col.slang` + GEMM (reuses mm kernel)
- [ ] `conv2d` with groups support
- [ ] `conv_transpose2d` via `col2im.slang`
- [ ] `conv2d_direct.slang` for small kernels (3×3, 1×1)

**Testing gate:**
- [ ] Conv2d matches `torch.nn.functional.conv2d` on CPU
- [ ] Various configurations: stride, padding, dilation, groups
- [ ] Depthwise convolution (groups=channels)

### 2.9 Normalization (Slang — `[Differentiable]`)

- [ ] `layer_norm`
- [ ] `group_norm`
- [ ] `batch_norm` (training + eval modes)

**Testing gate:**
- [ ] LayerNorm output has mean≈0, std≈1 along normalized dims
- [ ] BatchNorm running stats update correctly in training mode
- [ ] BatchNorm uses running stats in eval mode

### 2.10 Embedding & Indexing (Slang)

- [ ] `embedding` — lookup table
- [ ] `index_select`, `gather`, `scatter_`
- [ ] `index_put_`, `masked_fill_`

**Testing gate:**
- [ ] Embedding lookup matches CPU for random indices
- [ ] Out-of-bounds index produces error (not silent corruption)
- [ ] Scatter accumulate mode correct

**Stage 2 Deliverable:** ResNet-18 and GPT-2 124M inference correct on SwiftShader. CPU shader tests pass for all ops.

---

## Stage 3: Autograd & Backward Pass

**Goal:** `loss.backward()` works. Dramatically streamlined by Slang autodiff.

### 3.1 Tier 1 — Slang Autodiff Backward (majority of ops)

- [ ] Compile all `[Differentiable]` shaders with backward entry points
- [ ] Register backward kernels to `PrivateUse1`
- [ ] CPU shader tests: verify `bwd_diff` outputs vs numerical gradients
- [ ] Ops covered: all unary, binary, activations, softmax, layer_norm, group_norm, mse_loss, avg_pool2d, sum, mean

**Testing gate:**
- [ ] `torch.autograd.gradcheck()` passes for every Tier 1 op on SwiftShader
- [ ] CPU shader test: every `bwd_diff` output within 1e-3 of numerical gradient

### 3.2 Tier 2 — PyTorch Autograd Decomposition

- [ ] Verify `mm` backward works (decomposes to mm + transpose)
- [ ] Verify `bmm`, `addmm`, `linear` backward

**Testing gate:**
- [ ] `gradcheck()` passes for mm, bmm, addmm
- [ ] MLP backward produces correct weight gradients

### 3.3 Tier 3 — Hand-Written Slang Backward

- [ ] `convolution_backward` via col2im + GEMM (Slang)
- [ ] `max_pool2d_backward` using saved indices (Slang, `[BackwardDerivative(custom_fn)]`)
- [ ] `batch_norm_backward` training mode (`[BackwardDerivative(custom_fn)]`)
- [ ] `embedding_backward` scatter_add (Slang)
- [ ] `flash_attn_backward` recomputation-based (Slang with groupshared)

**Testing gate:**
- [ ] `gradcheck()` passes for each Tier 3 op
- [ ] Flash attention backward matches naive attention backward within tolerance

### 3.4 Custom Autograd Functions

- [ ] Flash attention as `torch::autograd::Function` (forward + backward registered on AutogradPrivateUse1)

### 3.5 End-to-End Backward Correctness

- [ ] Slang autodiff vs analytical gradients for gelu, softmax, layer_norm
- [ ] `opcheck()` for registration correctness on all ops
- [ ] MNIST MLP trains to convergence on SwiftShader
- [ ] Loss decreases monotonically for overfit-on-single-batch test

**Stage 3 Deliverable:** MNIST trains to convergence on SwiftShader.

---

## Stage 6: Advanced Operators

- [ ] Flash attention fwd+bwd (Slang with groupshared tiling)
- [ ] RoPE (Slang `[Differentiable]`)
- [ ] `interpolate`, `grid_sample` (Slang `[Differentiable]`)
- [ ] `sort`, `topk`, `cumsum`, `cumprod` (Slang)
- [ ] CPU fallback system for unimplemented ops

**Testing gate:**
- [ ] Flash attention matches `F.scaled_dot_product_attention` on CPU
- [ ] RoPE matches reference implementation
- [ ] CPU fallback: op not on Vulkan runs on CPU transparently

**Stage 6 Deliverable:** Verified models — ResNet-18/50, GPT-2, BERT, ViT, YOLO v8, Llama 2 7B (LoRA).
