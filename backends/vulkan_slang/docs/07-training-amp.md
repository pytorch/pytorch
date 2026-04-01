# Training Support: RNG, Optimizers & Mixed Precision

Covers Stages 4 (training loop support) and 5 (AMP & performance).

---

## Stage 4: Tensor Factories, RNG & Optimizer Support

**Goal:** Standard training loop with optimizers works on SwiftShader.

### Tensor Factories

- [ ] `empty` — allocate uninitialized tensor on Vulkan
- [ ] `zeros` — allocate + fill with 0
- [ ] `ones` — allocate + fill with 1
- [ ] `full` — allocate + fill with value
- [ ] `arange`, `linspace` — range tensors
- [ ] `eye` — identity matrix
- [ ] `rand`, `randn` — random tensors (requires RNG)
- [ ] `empty_like`, `zeros_like`, `ones_like`, `rand_like`, `randn_like`

**Testing gate:**
- [ ] Each factory produces correct shape, dtype, device
- [ ] `zeros` and `ones` values verified after round-trip to CPU
- [ ] `arange` matches `torch.arange` on CPU
- [ ] `eye` is identity matrix of correct size

### Random Number Generation

**Philox4x32-10 PRNG on GPU:**
- [ ] `shaders/random/philox.slang` — Philox PRNG module
- [ ] `VulkanGeneratorImpl` (C++) — manages Philox state, counter, key
- [ ] Integration with `torch.manual_seed()` and `torch.vulkan.manual_seed()`
- [ ] `shaders/random/dropout.slang` — fused mask generation + scale

**Random ops:**
- [ ] `uniform_` — fill tensor with uniform random [0, 1)
- [ ] `normal_` — fill tensor with normal random (Box-Muller via Philox)
- [ ] `bernoulli_` — Bernoulli samples
- [ ] `dropout` — fused Slang shader (mask + scale in one dispatch)

**Testing gate:**
- [ ] Seeded generator produces identical results across runs
- [ ] Two generators with same seed produce identical output
- [ ] `uniform_` values in [0, 1), mean ≈ 0.5 for large tensor
- [ ] `normal_` values: mean ≈ 0, std ≈ 1 for large tensor
- [ ] `dropout` zeros correct fraction of elements, scales survivors
- [ ] Dropout with p=0 is identity, p=1 is all zeros

### Optimizer Support Ops

These ops are needed by PyTorch optimizers (Adam, SGD, etc.):

- [ ] `addcmul_` — fused multiply-add (Adam)
- [ ] `addcdiv_` — fused divide-add (Adam)
- [ ] `lerp_` — linear interpolation (EMA)
- [ ] `clamp_` — gradient clipping
- [ ] `sqrt_`, `abs_`, `sign_` — in-place variants

**Testing gate:**
- [ ] Adam optimizer step matches CPU Adam step within tolerance
- [ ] SGD with momentum matches CPU SGD
- [ ] 10 Adam steps on a small MLP: weights diverge <1e-4 from CPU path

### Serialization

- [ ] `torch.save()` — Vulkan tensors → CPU → serialize
- [ ] `torch.load(map_location="vulkan:0")` — deserialize → CPU → Vulkan
- [ ] Checkpoint/resume: save model + optimizer state, reload, continue training

**Testing gate:**
- [ ] Save model on Vulkan, load on CPU: weights match
- [ ] Save on CPU, load on Vulkan: weights match
- [ ] Save model + optimizer, load, continue training: loss continues decreasing
- [ ] `map_location` routing works correctly

**Stage 4 Deliverable:** Full training pipeline on SwiftShader. Adam optimizer works. CIFAR-10 trains.

---

## Stage 5: Mixed Precision (AMP) & Performance

**Goal:** f16/bf16 training, competitive performance on real GPU.

### Autocast Policies

- [ ] Register policies on `AutocastPrivateUse1` dispatch key
- [ ] **f16 ops:** matmul, conv, linear, bmm, addmm
- [ ] **f32 ops:** norms, losses, softmax, reductions
- [ ] **Preserve dtype:** element-wise ops follow input dtype

### Slang Generics for f16

- [ ] All kernels use `<T : IFloat>` generics
- [ ] `compile_shaders.py` monomorphizes to both f32 and f16 SPIR-V
- [ ] Pipeline selection based on tensor dtype at dispatch time

### GradScaler

- [ ] `python/torch_vulkan/amp.py`: `VulkanGradScaler` (extends `torch.amp.GradScaler`)
- [ ] Inf/NaN check on Vulkan tensor (Slang reduce shader)
- [ ] Scale, unscale, update step

**Testing gate:**
- [ ] Autocast context manager promotes matmul inputs to f16
- [ ] Autocast preserves loss computation in f32
- [ ] GradScaler detects inf gradients and skips step
- [ ] AMP training loop converges on MNIST
- [ ] AMP training matches f32 training within reasonable tolerance

### Performance Optimization (Requires Real GPU)

- [ ] `mm_coopmat.slang` — KHR_cooperative_matrix GEMM
- [ ] Cooperative vectors (`CoopVec`) for small MLP layers — NVIDIA only
- [ ] Kernel fusion via Slang module composition
- [ ] Memory planning — pre-allocate peak memory
- [ ] Async host↔device transfers
- [ ] Subgroup reductions (replace workgroup reduce where possible)
- [ ] Pipeline cache warm-up on first run
- [ ] Auto-tune tile sizes per GPU (JSON config cache)

**Testing gate (real GPU only):**
- [ ] Tiled GEMM achieves >50% of theoretical peak GFLOPS on target GPU
- [ ] Cooperative matrix GEMM matches tiled GEMM output, faster for large sizes
- [ ] Memory pool avoids >90% of VMA allocations in steady-state training
- [ ] End-to-end training throughput within 2x of CUDA for ResNet-50

**Stage 5 Deliverable:** ResNet-50 and GPT-2 training with AMP on real GPU.

---

## FAQ

**Q: Do I need a GPU for Stage 4?**
A: No. All of Stage 4 runs on SwiftShader. RNG, optimizer ops, and serialization are all tested on CPU Vulkan.

**Q: What about bf16?**
A: bf16 requires `VK_KHR_shader_float16_int8` plus specific hardware support. Start with f16, add bf16 as a dtype variant via Slang generics when hardware support is confirmed.

**Q: Fused optimizer kernels?**
A: Stage 4 uses standard PyTorch optimizer decomposition (separate addcmul_, addcdiv_ calls). Fused Adam/SGD kernels are a Stage 5 performance optimization.

**Q: SwiftShader f16 support?**
A: Some SwiftShader builds lack f16 storage. Feature-gate AMP tests. The f16 shader code is still compiled and verified via Slang CPU target (Tier 1 tests).
