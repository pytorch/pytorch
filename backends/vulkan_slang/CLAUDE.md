# CLAUDE.md — PyTorch Vulkan Backend

Out-of-tree PyTorch backend: Vulkan compute + Slang shaders compiled to SPIR-V, registered via `PrivateUse1` as `torch.device("vulkan")`.
**All ops run on GPU (Vulkan compute)** — no CPU fallbacks. Ops not yet implemented raise `TORCH_CHECK(false, ...)`.
**All testing done on real GPU** (NVIDIA via native Vulkan). Do NOT set VK_ICD_FILENAMES — use the real GPU.

---

## Agent Behavior

**NEVER STOP:** Do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or summarize and wait. The human expects you to work autonomously until manually stopped.

- After completing one task, immediately move to the next unchecked item.
- When you finish writing code, immediately write tests, then move to the next feature.
- One-line status updates only — no multi-paragraph recaps.
- If blocked, raise TORCH_CHECK with descriptive error (no CPU fallbacks) or skip to next item.
- Regenerate `tools/generate_stub_shaders.py` after adding new shaders.
- Keep this file's checkboxes updated as you work.

---

## Environment

- **Python venv:** `.venv/` in project root (PyTorch 2.11.0+cu130). Created with `python3 -m venv --copies .venv`.
- **Slang:** Submodule at `third_party/slang` (v2025.8). Build with `cd third_party/slang && cmake --preset release && cmake --build --preset release`. Or download `slangc` binary from GitHub releases. Set `SLANGC=/tmp/bin/slangc` for shader compilation.
- **GPU testing:** Use real Vulkan GPU (NVIDIA). Do NOT set `VK_ICD_FILENAMES` — just run tests directly. SwiftShader path: `export VK_ICD_FILENAMES=/home/amit/swiftshader-build/build/Linux/vk_swiftshader_icd.json` (only for debugging without GPU).
- **Build:** `source .venv/bin/activate && pip install -e .` then `pytest tests/ --timeout=300 -p no:faulthandler`
- **Run single test:** `pytest tests/test_file.py::TestClass::test_name --timeout=30 -p no:faulthandler`
- **Note:** Must use `-p no:faulthandler` and redirect stderr (`2>/dev/null`) — Vulkan cleanup segfaults on exit corrupt output.
- **CPU shader tests:** `cd cpu_tests && mkdir -p build && cd build && cmake .. && make && ctest`

---

## Documentation Map

Read the relevant doc when starting work on that stage.

| Doc | Covers |
|-----|--------|
| [`docs/01-architecture.md`](docs/01-architecture.md) | Project structure, dependencies, coding standards |
| [`docs/02-vulkan-runtime.md`](docs/02-vulkan-runtime.md) | Context, Device, Memory, CommandBuffer, Stream, Pipeline |
| [`docs/03-slang-shaders.md`](docs/03-slang-shaders.md) | Slang shader patterns, compilation pipeline, autodiff |
| [`docs/04-pytorch-backend.md`](docs/04-pytorch-backend.md) | PrivateUse1 registration, DeviceGuard, Allocator, dispatch |
| [`docs/05-cpu-testing.md`](docs/05-cpu-testing.md) | SwiftShader setup, test tiers, CI config |
| [`docs/06-operators.md`](docs/06-operators.md) | All ops (forward + backward), model coverage |
| [`docs/07-training-amp.md`](docs/07-training-amp.md) | Factories, RNG, optimizers, AMP |
| [`docs/08-advanced.md`](docs/08-advanced.md) | Multi-GPU/DDP (Stage 7), torch.compile (Stage 8) |

---

## Current Test Status (2026-03-28): 1294 passed, 0 failed, 4 skipped (real GPU)

All ops run on GPU. Zero xfails — all previously-unimplemented ops are now on GPU.

| File | Passed | xfail |
|------|--------|-------|
| test_basic_ops | 10 | 0 |
| test_compute_ops | 36 | 0 |
| test_stage2_ops | 49 | 0 |
| test_stage2b_ops | 56 | 0 |
| test_stage3_autograd | 36 | 0 |
| test_stage4_training | 30 | 0 |
| test_stage5_amp | 4 | 0 |
| test_stage6_advanced | 23 | 0 |
| test_stage8_compile | 18 | 0 |
| test_model_ops | 47 | 0 |
| test_e2e_models | 30 | 0 |
| test_correctness | 281 | 0 |
| test_dtype_support | 139 | 0 |
| test_mnist_training | 42 | 0 |
| test_math_correctness | 492 | 0 |

### Key fixes applied
- **Vulkan cleanup segfault** (FIXED): Python `atexit.register(_c_ext._shutdown)` for proper singleton destruction ordering
- **Vulkan barrier validation** (FIXED): Removed `VK_ACCESS_HOST_READ_BIT` from compute shader barrier
- **"tensor does not have a device"** (FIXED): `@torch.compiler.disable` monkey-patches for F.conv2d/layer_norm/batch_norm/group_norm/linear/sdpa
- **Autograd backward** (FIXED): Analytical gradients + `at::native_*_backward` instead of nested `.backward()`
- **pow NaN** (FIXED): Shader handles negative bases with `pow(abs(base), exp)` + sign correction
- **group_norm** (FIXED): Multi-pass approach avoids shared-memory reduction; elementwise apply shader + existing GPU ops for mean/var
- **index.Tensor** (GPU): Advanced indexing with 1 or 2 index tensors via GPU shader
- **index_put_** (GPU): 1D indexed writes via GPU shader
- **1D padding** (GPU): New pad1d shader for 2-element pad on last dim
- **grid_sampler_2d** (GPU): Bilinear grid sampling shader with zeros/border/reflection padding
- **avg_pool2d** (FIXED): Added `count_include_pad` push constant flag to shader
- **torch.compile** (FIXED): `@torch.compiler.disable` prevents FakeTensor tracing into workaround patches
- **torch.compile autograd** (FIXED): Backward helper ops on PrivateUse1 enable AOT Autograd tracing
- **Generator/RNG** (FIXED): VulkanGeneratorImpl via getNewGenerator, torch.manual_seed propagation
- **_to_copy recursion** (AVOIDED): Do NOT register _to_copy on PrivateUse1 — causes infinite recursion with .cpu()
- **linear_backward 3D** (FIXED): Flatten batched tensors to 2D before mm in backward pass
- **mse_loss** (GPU): Full GPU shader for forward + backward (was CPU fallback)
- **arange overloads** (FIXED): Registered default + start overloads (not just start_step)
- **Profiler stubs** (DONE): No-op ProfilerStubs registered via registerPrivateUse1Methods
- **tiled matmul** (PERF): Shared-memory tiled mm_tiled.slang replaces naive per-element matmul
- **adaptive_avg_pool2d** (GPU): Proper GPU shader with ceil/floor window calculation (was CPU for non-divisible)
- **conv2d dilation** (GPU): Added dilation support to conv2d shader (was CPU fallback)
- **conv_transpose2d** (GPU): New GPU shader for transposed convolution
- **addmm general** (GPU): Uses GPU mul+add instead of CPU fallback for non-unit beta/alpha
- **SDPA** (GPU): Reimplemented using bmm+softmax composition (custom shader had precision issues on SwiftShader)
- **max_pool2d_backward** (GPU): Fixed race condition — gather-based backward replaces scatter-based
- **No CPU fallbacks**: All ops either run on GPU or raise TORCH_CHECK. No silent CPU computation.
- **Fused backward shaders** (PERF): sigmoid_backward, tanh_backward, silu_backward use single-pass Slang shaders (was 3-6 separate dispatches)
- **Fused softmax_backward** (PERF): Shared-memory dot product + elementwise in one workgroup per row (was 4 dispatches)
- **Fused layer_norm_backward** (PERF): 2 dispatches (grad_input + grad_weight/bias) via shared-memory reduction shaders (was 22 dispatches decomposed)
- **Fused group_norm_backward** (PERF): 2 dispatches via shared-memory reduction with per-group weight expansion (was ~25 dispatches decomposed)
- **Fused batch_norm_backward** (PERF): 2 dispatches (one workgroup per channel) via shared-memory reduction (was ~20 dispatches decomposed)
- **Per-dim reduction shaders** (PERF): sum_dim, max_dim, min_dim, prod_dim — single GPU dispatch per reduction (was O(rows) host-loop dispatches)
- **GPU buffer copy** (PERF): In-place ops use GPU copy shader instead of host read+write roundtrip
- **Scalar ops cleanup** (PERF): Replaced at::full_like + vulkan_mul patterns with vulkan_mul_scalar throughout (eliminates temp tensor allocation)
- **addmm CPU fix** (FIXED): General-case addmm was using at::mul/at::add (CPU dispatch) — replaced with vulkan_mul_scalar/vulkan_add
- **Scalar binary shaders** (PERF): add_scalar, sub_scalar, mul_scalar, div_scalar use push constant (no temp tensor allocation)
- **Scalar comparison shaders** (PERF): eq/ne/lt/gt/le/ge scalar variants use push constant (eliminates full_like allocation)
- **Fused threshold_backward** (PERF): Single-pass shader (was gt + to_float + mul = 3 dispatches)
- **RNG scaling on GPU** (FIXED): uniform_/normal_ now scale via GPU ops (was CPU host roundtrip)
- **BCE loss** (GPU): Binary cross entropy forward + backward via Slang shaders
- **BCE with logits** (GPU): Numerically stable BCE with logits via Slang shader
- **smooth_l1_loss / huber_loss** (GPU): Forward + backward with configurable beta/delta
- **hardtanh / relu6** (GPU): Forward + backward + in-place via Slang shaders
- **hardswish** (GPU): Forward + backward + in-place via Slang shaders
- **hardsigmoid** (GPU): Forward + backward + in-place via Slang shaders
- **softplus** (GPU): Forward + backward with configurable beta/threshold via Slang shaders
- **instance_norm** (GPU): Via group_norm(num_groups=C) Python redirect
- **F.normalize** (GPU): Via norm.ScalarOpt_dim + clamp_min registration
- **Dropout2d** (GPU): Via bernoulli_ registration (uniform RNG + threshold)
- **clamp_min/clamp_max** (GPU): Wrapper over existing clamp shader
- **bernoulli_** (GPU): Float and tensor probability variants via Philox RNG
- **Comparison ops GPU bool** (PERF): float→bool conversion via GPU shader (was CPU read+loop+write roundtrip)
- **AMP unscale GPU** (PERF): Fused inf/nan check + unscale in single GPU shader per grad (was CPU copy+check+mul+copy per grad)
- **bernoulli_ bool copy GPU** (PERF): Bool buffer copy via GPU shader (was CPU read+write roundtrip)
- **Conv1d** (GPU): Via Python monkey-patch unsqueeze→conv2d→squeeze (no separate shader needed)
- **Math ops** (GPU): tan, atan, atan2, log2, log10, log1p via Slang shaders
- **Check ops** (GPU): isnan, isinf via Slang shaders (isfinite via composition)
- **KL Divergence** (GPU): Forward with log_target support via Slang shader
- **L1 Loss** (GPU): Decomposes via PyTorch into abs(self-target), all ops on GPU
- **Mish activation** (GPU): Forward + backward via Slang shaders (x * tanh(softplus(x)))
- **cosine_similarity** (GPU): Via clamp_min.Tensor + clamp_min.Tensor_out registration
- **clamp_min_** (GPU): In-place variant for cosine_similarity decomposition
- **RMSNorm** (GPU): Fused single-pass forward shader + 2-pass backward (grad_input + grad_weight), custom autograd wrapper, Python API `torch_vulkan.rms_norm()`
- **nll_loss ignore_index** (GPU): Forward + backward shaders handle `ignore_index=-100` for LLM padding tokens
- **cross_entropy backward** (FIXED): Removed PrivateUse1 registration — PyTorch decomposes via autograd into `log_softmax` + `nll_loss` for proper backward
- **Shape ops autograd** (FIXED): All shape ops (view, reshape, transpose, permute, unsqueeze, squeeze, expand, select, slice) have autograd wrappers on `AutogradPrivateUse1` — backward = inverse shape transform
- **BLAS autograd** (FIXED): mm/bmm/addmm/linear NOT registered on AutogradPrivateUse1 (breaks torch.compile). PyTorch's built-in autograd handles backward since shape ops now have autograd.
- **Qwen3 training** (VERIFIED): End-to-end training step: Embedding → RMSNorm → GQA Attention → SwiGLU MLP → cross_entropy → backward → SGD step — all on GPU, gradients match CPU
- **Large-vocab softmax** (FIXED): softmax/log_softmax decompose into amax+exp+sum+div for rows > 256 (enables 151936-class cross_entropy)
- **Tied embeddings** (VERIFIED): Weight tying works via `tie_weights()` after `.vulkan()` — `.vulkan()` breaks ties (opaque allocator), must re-establish
- **Gradient checkpointing** (VERIFIED): `torch.utils.checkpoint.checkpoint()` works with Vulkan tensors (non-reentrant mode)
- **Gradient clipping** (VERIFIED): `clip_grad_norm_` + `clip_grad_value_` work on Vulkan params
- **clamp_max_/clamp_min.out/clamp_max.out** (GPU): Added for `_foreach_clamp_*` support (gradient clipping)
- **Fused SwiGLU** (PERF): Single-pass `silu(gate) * up` Slang shader + backward, Python API `torch_vulkan.swiglu()`
- **f16/bf16 dtype support** (GPU): Widen-compute-narrow approach — f16/bf16 tensors stored natively, GPU cast shaders (uint32 bit manipulation) convert to f32 for compute, cast back. Works on SwiftShader (no VK_KHR_shader_float16_int8 needed). All ops: unary, binary, matmul, shape, activation, reduction, normalization, loss, backward.
- **Cast shaders** (GPU): `cast_f16_to_f32`, `cast_f32_to_f16`, `cast_bf16_to_f32`, `cast_f32_to_bf16` — uint32 bit manipulation, f16 pairs packed into uint32
- **AMP autocast policies** (GPU): mm/bmm/addmm/linear/conv → lower_precision_fp (f16); layer_norm/softmax/nll_loss → fp32
- **bitwise_and.Tensor_out** (GPU): For bool tensors via mul (needed for tensor repr/isfinite)
- **random_.from** (GPU): Fill with random integers via uniform + scale + floor
- **max.dim / min.dim** (GPU): Returns (values, indices) via amax+argmax / amin+argmin composition
- **slice/select backward** (FIXED): Opaque allocator makes slice/select return copies, not views. Backward was using `grad.slice(...).copy_(go)` which wrote to detached copy. Fixed: build grad on CPU (where slice is a view), then move to Vulkan.
- **narrow registration** (FIXED): `vulkan_narrow_adapter` existed but was never connected with `m.impl("narrow", ...)`. Missing narrow broke `cat` backward (autograd decomposes cat backward into narrow).
- **chunk registration** (AVOIDED): Do NOT register `chunk` on PrivateUse1 — it's CompositeImplicitAutograd and decomposes into slice ops. Registration breaks autograd ("derivative not implemented").
- **layer_norm large norm_size** (FIXED): Decompose into mean+var+normalize+scale+bias for norm_size > 256 (fused shader limited by workgroup size)
- **masked_fill_.Scalar** (GPU): Registered on PrivateUse1 for causal attention masks
- **kl_div_backward** (GPU): Registered on PrivateUse1 for KL divergence loss backward
- **MNIST convergence tests** (42 tests): 7 model architectures × (fp32 + fp16 + bf16 + vs_cpu + AMP), all converging on Vulkan
- **Deferred execution** (PERF): Command buffer batching — multiple GPU dispatches recorded into a single command buffer, submitted on flush. Reduces submit+fence overhead from ~0.1ms/dispatch to amortized ~0.02ms/dispatch.
- **Buffer quarantine** (PERF): Freed buffers quarantined until command buffer completes, then recycled. Eliminates WAR hazard flushes (was 300+ flushes per training step, now ~15).
- **Transpose-aware matmul** (PERF): bmm/mm shaders accept `transpose_a`/`transpose_b` flags, avoiding CPU roundtrip for `.contiguous()` on transposed views. Critical for attention (q@k.T) and linear (input@weight.T).
- **GPU strided copy** (PERF): `copy_strided_copy_fwd` shader copies non-contiguous→contiguous on GPU (was CPU readback roundtrip). Supports up to 5D tensors.
- **Perf counters** (DEBUG): `_get_dispatch_count()`, `_get_flush_count()`, `_get_preread_flush_count()` etc. exposed via Python for profiling.
- **masked_scatter** (GPU): Vision embedding injection for Qwen2-VL — GPU shader with CPU-built index map
- **Conv3d** (GPU): Decomposed into temporal slices + Conv2d accumulation. Supports Qwen2-VL patch embedding `[2,14,14]` kernels.
- **unbind/outer** (DECOMPOSE): Work via existing select/unsqueeze+mul ops, no new registration needed
- **FP8 dtype support** (GPU): `float8_e4m3fn` and `float8_e5m2` via widen-compute-narrow (same approach as f16/bf16). GPU cast shaders: `cast_e4m3fn_to_f32`, `cast_f32_to_e4m3fn`, `cast_e5m2_to_f32`, `cast_f32_to_e5m2`. 4 FP8 values packed per uint32. All ops work automatically through ensure_float32()/cast_from_float32() path.
- **Fused SGD shader** (PERF): Single-dispatch SGD with momentum/weight_decay/nesterov — 1 dispatch per param instead of 13 (saves ~500 dispatches per training step). Python API `torch_vulkan.SGD()`.
- **nll_loss GPU-only** (PERF): count_valid_targets uses dedicated `nll_loss_count` shader (asint comparison on int32-as-float target). total_weight stays as GPU scalar tensor — backward reads it from buffer binding, no CPU roundtrip. Eliminates 3 pre-read flushes in CE loss path.
- **Buffer-specific pre-read flush** (PERF): `VulkanBuffer::read()` only flushes if THIS buffer has pending GPU work (via `is_buffer_in_flight()` callback). Eliminates unnecessary flushes when reading newly allocated or CPU-only buffers.
- **GPU int64→int32 conversion** (PERF): `i64_to_i32.slang` shader extracts low 32 bits from int64 (stored as uint32 pairs). Avoids CPU roundtrip for embedding/nll_loss target indices.
- **`_flush()` Python API** (DEBUG): Exposes `flush_stream()` for benchmarking synchronization.
- **Zero-copy view/reshape** (PERF): `vulkan_view` creates a new TensorImpl sharing the same Storage (zero dispatches, no GPU copy). All view/reshape/unsqueeze/squeeze ops are now free.
- **Contiguous returns self** (PERF): `vulkan_contiguous` returns `self` since all Vulkan tensors from opaque allocator are already contiguous. Eliminates clone dispatch on every `.contiguous()` call.
- **Removed F.linear monkey-patch** (PERF): `F.linear` no longer injects zeros bias — our registered `vulkan_linear` handles None bias natively. Saves ~5 dispatches per linear call (7 calls/layer).
- **vulkan_linear skip reshape** (PERF): For 2D input with no bias, linear is just 1 mm dispatch (was 10 with patch + reshape overhead).
- **CPU scalar tensor shortcut** (PERF): `mul.Tensor`, `add.Tensor`, `sub.Tensor`, `div.Tensor` detect CPU 0D scalar operands and redirect to scalar shader variants (1 dispatch instead of 3 = device-copy + expand + op).
- **Same-shape binary op skip** (PERF): `binary_op` skips `.expand().contiguous()` when both tensors already have the same shape (1 dispatch instead of 3).
- **Atomic embedding_dense_backward** (PERF): CAS-based float atomics via `InterlockedCompareExchange` loop — no CPU readback for duplicate index check. GPU i64→i32 conversion + atomic accumulation in single dispatch (was CPU flush + readback + histogram).
- **Fused log_softmax_backward** (PERF): Single-workgroup shader with shared-memory reduction (1 dispatch vs 5 for composition)
- **movedim skip optimization** (PERF): softmax/log_softmax/reductions skip `movedim(dim, -1)` when dim is already last — saves 1-2 dispatches per call
- **upsample_nearest2d_backward** (GPU): Gather-based backward for UNet/diffusion models — each input pixel accumulates from output region
- **FP8 autocast** (GPU): `float8_e4m3fn`/`float8_e5m2` work with `torch.autocast("vulkan")` and `_scaled_mm` for explicit quantized matmul
- **Fused large-row softmax** (PERF): Single-dispatch softmax/log_softmax forward+backward for any row size (was 7-8 dispatches for rows > 256). Workgroup-per-row with strided thread access + shared memory reduction.
- **Fused large-row softmax_backward** (PERF): Single-dispatch backward for any row size (was 4-5 dispatches for rows > 256)
- **upsample_bilinear2d_backward** (GPU): Gather-based backward for diffusion/UNet models with bilinear upsampling
- **adaptive_avg_pool2d_backward** (GPU): Custom AutogradPrivateUse1 wrapper + GPU shader for per-input-pixel gradient gathering. Fixes gradient propagation through AdaptiveAvgPool2d (was silently returning None).
- **conv_backward f16/bf16** (FIXED): CPU fallback convolution backward now converts to f32 before computing (CPU doesn't support f16/bf16 well), converts back after.
- **Fused add_rms_norm** (PERF): Single-dispatch `h_new = residual + shortcut; normed = weight * rms_norm(h_new)`. Saves 1 dispatch vs separate add+rms_norm per call. Python API `torch_vulkan.add_rms_norm(residual, shortcut, weight)` → `(normed, h_new)`. Backward: fused `rms_norm_backward + add_grad` in single workgroup pass (2 dispatches instead of 3). Saves 7 fwd + 7 bwd dispatches in 4L model.
- **scaled_bmm** (PERF): `scale * (q @ k.T)` fused into single dispatch via scale param in bmm shader. Saves 1 dispatch per attention layer. Python API `torch_vulkan.scaled_bmm(q, k, scale)`.
- **Flash Attention** (PERF): Fused `QK^T + softmax + @V` in a single dispatch per (b,h,q) workgroup. Eliminates the intermediate `[B*H, N, S]` attention weight matrix. Saves 12 forward + 4 backward dispatches in 4L model (76→64 fwd, 142→138 bwd). Handles GQA natively (no K/V expansion needed). Python API `torch_vulkan.flash_attention(Q, K, V, scale, is_causal)`. Backward: 2 dispatches (grad_Q + grad_KV; D_i computed inline). Numerically identical to F.scaled_dot_product_attention (max diff <1e-6).
- **Seq-major attention layout** (PERF): Auto-detect `view+transpose` pattern for Q/K/V (stride(1)==D is the key signal). When detected, all 3 attention shaders (fwd, bwd_Q, bwd_KV) read Q/K/V/grad_out in native seq-major `[B,S,H,D]` layout without `.contiguous()` copies. Forward output also written in seq-major `[B,S,H,D]` storage returned as non-contiguous `[B,H,S,D]` view — enables zero-copy `transpose(1,2).reshape(B,S,-1)` on output. Backward `grad_out` also detected as seq-major when it flows back through the `transpose+reshape` chain. Net: -16 fwd (-4 from output reshape + -12 from no contiguous copies) and -16 bwd dispatches in 4L model (64→48 fwd, 126→110 bwd). go_seq_major tracked separately from q_seq_major in BwdParams.
- **Inline D_i fusion** (PERF): Eliminated separate `flash_attention_di` shader dispatch. `bwd` shader computes `di_val = dot(go[q], fwd_out[q])` via shared-memory reduction before the k-loop. `bwd_kv` shader computes `go_dot_out` fused with `go_dot_v` in the same element loop. Backward: 3 dispatches → 2 dispatches. Saves 4 bwd dispatches in 4L model (110→106 bwd).
- **Inplace binary op dirty-buffer bug** (BUG FIX): `_foreach_mul_` + `_foreach_add_` (used by SGD momentum) dispatched inplace shaders as `{self, other}`. Smart barrier code marks last `num_outputs=1` tensor as dirty → marked `other` dirty, not `self`. Next dispatch reading `self` skipped the barrier and saw the old pre-write value. Fixed: pass `num_outputs=2` for all tensor inplace dispatches (`add_`, `sub_`, `mul_`, `div_`), marking `self` dirty. SGD momentum convergence error was 4.2e-3 (vs 1e-3 tolerance), now 0.
- **Optimizer batch dirty-buffer bug** (BUG FIX): SGD batch and AdamW batch dispatches used interleaved buffer layouts (e.g., `{p0, g0, p1, g1, ..., pN, gN}` for SGD batch) but passed `num_outputs=n` — which marks only the LAST n entries dirty, missing params at even indices 0, 2, ... Fixed: pass `num_outputs=bufs.size()` (mark all dirty), which is over-conservative but correct. Also fixed SGD no-momentum single dispatch (`{param, grad}` with `num_outputs=2` instead of default 1).
- **Wave-intrinsic flash attention** (PERF): For D<=32, all 3 attention shaders (fwd, bwd_Q, bwd_KV) dispatch 32 threads/workgroup and use `WaveActiveSum` for dot-product reductions (zero barrier syncs per key position). With D=32 the standard 256-thread path wastes 224 threads and executes 8 `GroupMemoryBarrierWithGroupSync` calls per key; the wave path uses 0 barriers. New shaders: `flash_attention_fwd_wave.slang`, `flash_attention_bwd_wave.slang`, `flash_attention_bwd_kv_wave.slang`. Dispatch routing in `attention_ops.cpp` selects wave path when `D <= 32`. Numerically identical to standard path (max diff <2e-6). Speedup on RTX 4060 Ti (D=32, B=2, 4L model): fwd 5.19ms→3.39ms (-35%), bwd 14.37ms→6.84ms (-52%), total ~20ms→~11ms (-47%).
- **D<=64 flash attention fwd+bwd** (PERF): For 32 < D <= 64 (Qwen3-0.6B head_dim=64), all 3 attention shaders use 64-thread d64 variants (6 barrier syncs per reduction vs 8 for 256-thread standard). New shaders: `flash_attention_fwd_d64.slang` (pre-existing), `flash_attention_bwd_d64.slang`, `flash_attention_bwd_kv_d64.slang`. Dispatch routing: D<=32 → wave (0 barriers/reduction), D<=64 → d64 (6 barriers), D>64 → standard (8 barriers). Numerically identical (max diff <1e-6).
- **Batched SGD shader** (PERF): `sgd_batch.slang` processes up to 7 parameters per dispatch using 2D grid (Y=param_index). Per-param config `{numel, lr, weight_decay}` passed via push constants (88 bytes total). `torch_vulkan.SGD.step()` uses batch path for no-momentum f32 params. Optimizer step: 39 → 6 dispatches for 39 params (saves 33 dispatches, 0.52ms vs 0.91ms). Python API `_c_ext._sgd_batch_step(params, grads, lr, weight_decay)`.
- **Batched SGD15 shader** (PERF): `sgd_batch15.slang` processes up to 15 parameters per dispatch (30 bindings, 184-byte push constants — NVIDIA supports 256 bytes). MAX_BINDINGS increased to 32. `torch_vulkan.SGD.step()` uses 15-param batch for n>7, falls back to 7-param batch for n<=7. Optimizer step: 36 params → 3 dispatches (was 6). Python side sets BATCH=15 automatically.
- **Batched AdamW shader** (PERF): `adamw_batch.slang` processes up to 3 parameters per dispatch (4 bindings/param: param+grad+m+v). Per-param config `{numel, lr, beta1, beta2, eps, wd, bc1, bc2}` passed via push constants (100 bytes total). `torch_vulkan.AdamW.step()` uses batch path for f32 params. Optimizer step: 39 → 13 dispatches for 39 params (saves 26 dispatches). Python API `_c_ext._adamw_batch_step(params, grads, m_bufs, v_bufs, lr, beta1, beta2, eps, wd, bc1, bc2)`.
- **Batched AdamW7 shader** (PERF): `adamw_batch7.slang` processes up to 7 parameters per dispatch (28 bindings, 228-byte push constants — within NVIDIA's 256-byte guarantee). `vulkan_adamw_batch_step()` dispatches to batch7 for n>3, falls back to batch3 for n<=3. `torch_vulkan.AdamW.step()` uses BATCH=7. Optimizer step: 39 → 6 dispatches for 39 params (was 13 with BATCH=3, saves 7 more dispatches).
- **241 Slang shaders** compiled (was 239)
- **Fused nll_loss_mean** (PERF): Single-pass workgroup shader combines per-sample loss + valid-sample count + partial sum reduction. CE: 7→4 dispatches, nll_loss mean: 6→3 dispatches for N≤256 batches.
- **Broadcast-aware add** (PERF): `binary_add_broadcast` shader uses `b[i % numel_b]` for suffix-broadcast patterns like `[B,H,S,S]+[S,S]` (causal mask add in attention). 1 dispatch instead of 2 (eliminates expand). Saves 4 dispatches per forward pass in 4-layer model.
- **True in-place add_/sub_/mul_/div_** (PERF): `binary_*_inplace` shaders use `RWStructuredBuffer` to write directly to `self` without intermediate allocation. `vulkan_add_/sub_/mul_/div_` (f32 same-shape) = 1 dispatch instead of 2 (eliminates dispatch_copy_buffer). Saves ~62 dispatches per backward pass in L=2 model.
- **True in-place add_/sub_/mul_/div_ scalar** (PERF): `binary_*_scalar_inplace` shaders with single `RWStructuredBuffer` binding for `self`. Saves 1 dispatch per scalar inplace op for f32 contiguous tensors when called via `torch.ops.aten.add_.Scalar` (Python-level `a.add_(scalar)` still goes through ADInplaceOrView key = 2 dispatches; aten-level = 1).
- **Fused layer_norm forward** (PERF): Modified `layer_norm.slang` to output per-row `mean` and `rstd` via bindings 4/5. Eliminated 7 redundant pre-dispatch GPU ops (mean, x^2, mean_x^2, mean^2, var, var+eps, rsqrt) that were pre-computing values the fused shader recomputed anyway. LayerNorm forward: 8 dispatches → 1 dispatch.
- **Norm backward empty buffers** (PERF): Replaced `at::zeros` with `at::empty` for `grad_weight`/`grad_bias` output buffers in `layer_norm_backward`, `group_norm_backward`, and `batch_norm_backward`. The weight/bias backward shaders fully initialize their outputs (non-atomic assignment, not accumulation), so pre-zeroing is unnecessary. Saves 2 dispatches per normalization backward call (1 per zero allocation).
- **RMSNormGated** (GPU): Fused `weight * rms_norm(input) * silu(gate)` for Qwen3.5-0.8B GatedDeltaNet layers. Forward + backward Slang shaders. Python API `torch_vulkan.rms_norm_gated()`.
- **bf16 embedding lookup** (PERF): Large-vocab bf16 embeddings (248320×1024=485MB) skip f32 upcast via raw uint32 copy shader. Avoids OOM on SwiftShader's ~500MB buffer limit.
- **Chunked LM head** (FIX): `vulkan_linear` automatically chunks weight along out_features when `out_features * in_features > 16.7M` (65535×256 workgroup limit). Each chunk is read from GPU via `VulkanBuffer::read()` + staged to a fresh Vulkan tensor. Fixes OOM for 248320-vocab LM head (`[4,1024] @ [248320,1024].T` → no 970MB f32 weight upcast). Also prevents Vulkan validation "groupCountX exceeds limit" errors.
- **Chunked mm backward** (FIX): `vulkan_mm` automatically chunks along K dimension when either matrix numel > 16.7M. Enables `grad_input = grad_output[4,248320] @ weight[248320,1024]` backward via K-split accumulation. Both forward (vulkan_linear chunking) and backward (vulkan_mm K-chunking) now work for 248k-vocab LM head.
- **Qwen3.5-0.8B support** (VERIFIED): GatedDeltaNet torch fallback (all constituent ops on GPU), causal conv1d, partial mRoPE (partial_rotary_factor=0.25), RMSNormGated, large-vocab bf16 embedding, chunked LM head forward+backward — all verified on Vulkan.
- **Int32 buffer reinterpret for indices** (PERF): When int32 index tensors (embedding lookup, NLL loss targets) are already on Vulkan, skip GPU copy to float buffer — create a TensorImpl sharing the same Storage but with float dtype. Shaders use `asint()` to read int32 from float-typed binding. Saves 1 dispatch per embedding fwd, 1 per embedding bwd, 1 per NLL loss.
- **Smart barrier insertion** (PERF): Global memory barrier (after every dispatch) replaced with dependency-aware barriers. `dispatch_shader` tracks which buffers are written (last `num_outputs` tensors) in a `dirty_buffers` set. Barrier only emitted when the next dispatch reads a buffer in `dirty_buffers`. 37.5% barrier skip rate on Qwen3-style training step (45/120 barriers skipped). New perf counters: `_get_barrier_count()`, `_get_barrier_skip_count()`. Multi-output shaders annotated with `num_outputs` parameter.
- **triu/tril bounds check** (BUG FIX): Shader dispatches `ceil(numel/256)` workgroups × 256 threads; threads beyond `numel` had no bounds check and wrote out-of-bounds into adjacent VMA-allocated buffers. Fixed: added `numel` push constant and `if (tid.x >= params.numel) return;` guard in both `triu.slang` and `tril.slang`.
- **Zero-copy N-D transpose** (PERF): `vulkan_transpose` for float32 returns metadata-only view with swapped strides (no GPU copy). `vulkan_bmm` detects last-2-dims-transposed views via `is_last2_transposed()` and routes to `vulkan_bmm_ex` with `transpose_b=true`. Saves 1 dispatch per `q@k.T` (attention score computation).
- **unsqueeze/squeeze direct-stride** (BUG FIX): `vulkan_unsqueeze`/`vulkan_squeeze`/`vulkan_squeeze_dim` rewrote to directly insert/remove strides from existing stride array instead of calling `vulkan_view`. `at::detail::computeStride` gave wrong batch strides for non-contiguous transposed inputs, causing GQA k-vector mismatch (0.06 max diff in 93% of values).
- **expand contiguous guard** (BUG FIX): `vulkan_expand` now calls `self.contiguous()` for non-contiguous inputs before passing to GPU shader. The `copy_expand_fwd` shader computes input offsets from contiguous strides and reads wrong memory for zero-copy transposed/unsqueezed views.
- **In-stride-aware expand** (PERF): `expand.slang` accepts `in_strides[8]` push constant, letting `vulkan_expand` skip pre-`contiguous()` for f32 inputs. Non-contiguous inputs (GQA k/v after unsqueeze+transpose) now expand directly from their actual strides. Saves 1 dispatch per GQA k/v expand (8 dispatches saved per 4L forward pass).
- **Multi-tensor cat_n shader** (PERF): `cat_n.slang` handles 2–8 inputs in 1 dispatch (9 bindings: in0..in7 + out), vs N-1 pairwise dispatches. `vulkan_cat` uses single-dispatch path for ≤8 inputs. `MAX_BINDINGS` increased to 16 in both `dispatch.cpp` and `DescriptorSet.cpp`.

---

## PyTorch 2.11 PrivateUse1 Notes

- Many ops use `c10::SymInt`/`c10::SymIntArrayRef` — adapters in `Registration.cpp` using `symint_to_int()`.
- **Tensor? None args:** Python monkey-patching in `__init__.py` converts None bias/weight/mask to zero/identity tensors for Vulkan. Uses `@torch.compiler.disable` to avoid torch.compile issues.
- `convolution_overrideable` is the correct dispatch point for conv, NOT `conv2d`. `convolution_backward_overrideable` registered as CPU fallback.
- Event/Stream support implemented in `DeviceGuard.cpp` — single-stream backend, events are no-ops.
- **Int64 on Vulkan:** DO NOT reshape/view int64 tensors on Vulkan — corrupts data (8-byte int reinterpreted as 4-byte float). Move to CPU first.
- **SwiftShader Int64:** No SPIR-V Int64 support. Use 32-bit math only (see `mulhi32()` in `philox.slang`).
- **save_for_backward:** All tensor inputs needing gradients MUST be saved. Missing bias in save causes None grad.
- **Shader recompilation:** `SLANGC=/tmp/bin/slangc python3 tools/compile_shaders.py`. Never run `generate_stub_shaders.py` after — it overwrites real SPIR-V.
- **Testing:** Use `-p no:faulthandler` and `2>/dev/null` when running pytest. Cleanup segfault handled via Python atexit shutdown.
- **_to_copy:** Do NOT register `_to_copy` on PrivateUse1 — it causes infinite recursion (`.cpu()` dispatches through `_to_copy`). Use `copy_` and `_copy_from` instead.
- **Build:** Install `ninja` (`pip install ninja`) for parallel builds. Uses `MAX_JOBS=$(nproc)` automatically.
- **Fast rebuild:** `bash tools/rebuild.sh` — incremental ninja build (~3s no-change, ~20s single-file vs ~100s full rebuild).
- **slangc:** Available via `third_party/slang` submodule (v2025.8) or downloaded to `/tmp/bin/slangc`. Set `SLANGC=/tmp/bin/slangc` or build from submodule.
- **No CPU fallbacks:** All ops either dispatch to GPU Slang shaders or raise `TORCH_CHECK(false, ...)`. Never fall back to CPU silently.

---

## Remaining Work — Revised Plan (2026-03-26)

### Phase 1: Fix torch.compile (P0) — DONE

- [x] Create `csrc/ops/backward_ops.cpp` — all backward helper ops
- [x] Register backward ops on PrivateUse1 in Registration.cpp
- [x] Remove AutogradPrivateUse1 for standard ops, keep only: max_pool2d, convolution, sdpa, prelu, selu, clamp
- [x] Register Meta kernels for all backward helper ops + scalar ops
- [x] Remove xfail from test_compile_chain, add 6 more compile tests
- [x] All 259 tests pass (backward correctness verified)

### Phase 2: Backend Infrastructure (P0–P1) — DONE

- [x] Manual `tensor.vulkan()`, `module.vulkan()`, `is_vulkan` property in `__init__.py`
- [x] `torch._register_device_module("vulkan", VulkanModule)` with AMP APIs
- [x] `GeneratorImpl` + `getNewGenerator`/`getDefaultGenerator` in HooksInterface
- [x] `torch.manual_seed()` propagation + `torch.Generator(device="vulkan")` working
- [x] Add mandatory ops: `as_strided`, `resize_`

### Phase 3: Model Coverage Ops (P1) — DONE

- [x] `triu` / `tril` — causal attention masks
- [x] `F.pad` / `constant_pad_nd` — 1D and 2D padding on GPU
- [x] `index.Tensor` — advanced indexing (1 or 2 index tensors) on GPU
- [x] `repeat` / `repeat_interleave` — positional encodings, KV cache
- [x] `stack` — data collation
- [x] `erf` / `erf_` — GELU exact mode
- [x] `flip` / `roll` — tensor manipulation
- [x] `_unsafe_view` — internal view aliasing

### Phase 4: AMP & Autocast (P2) — DONE

- [x] `AutocastPrivateUse1` fallthrough registration — `torch.autocast("vulkan")` works
- [x] AMP functions on `torch.vulkan` module (`get_amp_supported_dtype`, etc.)
- [x] `_amp_foreach_non_finite_check_and_unscale_` + `_amp_update_scale_` for GradScaler
- [x] `reciprocal`, `sin`, `cos`, `logical_not`, `bitwise_not` ops
- [x] f16/bf16 dtype support via widen-compute-narrow (GPU cast shaders, all ops work)
- [x] Autocast policies: mm/bmm/addmm/linear/conv → lower_precision_fp; layer_norm/softmax → fp32
- [x] FP8 dtype support (float8_e4m3fn, float8_e5m2) via widen-compute-narrow (GPU cast shaders)
- [ ] f16 native compute shaders for compute-bound ops (mm, conv, attention) — future work

### Phase 5: All Ops on GPU — No CPU Fallbacks (P2) — DONE

All ops run on GPU via Slang shaders (198 compiled). Zero TORCH_CHECK stubs remaining.
- [x] Shape ops: permute, expand, cat, select, slice, flip, roll, repeat, pad (1D+2D), triu/tril
- [x] Unary ops: sin, cos, erf, reciprocal, logical_not, bitwise_not
- [x] Binary ops: fmod, remainder
- [x] Reductions: argmax/argmin, any/all, cumprod, norm (L1/L2)
- [x] Indexing: gather, scatter_, index.Tensor, index_put_, upsample_nearest2d
- [x] Sort/topk: per-row insertion sort + partial sort
- [x] Loss: mse_loss, bce, bce_with_logits, smooth_l1, huber, l1_loss, kl_div — forward + backward
- [x] Activations: hardtanh/relu6, hardswish, hardsigmoid, softplus, mish — forward + backward + in-place
- [x] Backward: all backward ops on GPU (gelu, elu, leaky_relu, selu, avg_pool2d, max_pool2d, layer_norm, batch_norm, group_norm, embedding, SDPA, PReLU, RoPE, hardtanh, hardswish, hardsigmoid, softplus, mish)
- [x] Pooling: adaptive_avg_pool2d, avg_pool2d_backward (GPU), max_pool2d_backward (GPU, gather-based)
- [x] Conv: conv1d (via conv2d), conv2d with dilation, conv_transpose2d
- [x] Interpolation: upsample_bilinear2d, grid_sampler_2d
- [x] Attention: SDPA via bmm+softmax composition, RoPE forward+backward
- [x] Normalization: group_norm + backward (multi-pass, no shared memory), instance_norm (via group_norm)
- [x] Training: bernoulli_, Dropout2d, F.normalize, clamp_min/clamp_max, cosine_similarity
- [x] Math ops: tan, atan, atan2, log2, log10, log1p
- [x] Check ops: isnan, isinf
- [ ] f16 native compute shaders for compute-bound ops (mm, conv, attention) — future work

### Phase 6: Advanced (P3)

- [x] Profiler stubs — `registerPrivateUse1Methods(&stubs)` for `torch.profiler`
- [ ] Stage 7 DDP — requires multi-GPU hardware, deprioritize
- [ ] torch.compile Inductor backend — kernel fusion (stretch goal)
- [x] CI hardening (Stage 1.2/1.3)

---

## Target: Train Qwen3-0.6B (Qwen/Qwen3-0.6B)

**Architecture:** 28-layer decoder-only transformer, 1024 hidden, 16 query heads / 8 KV heads (GQA), 128 head_dim, 3072 intermediate (SwiGLU), 151936 vocab, RMSNorm, RoPE (theta=1M), bfloat16.

### Gap Analysis: What's Missing

**Current state:** The backend trains CNNs and small transformers (tests pass for ResNet, small GPT-2-style models). Cross-entropy loss backward currently runs on CPU (tests move logits to CPU for loss). The following gaps block full Qwen3 training on Vulkan.

#### P0 — Blocking (training loop won't run without these)

1. **RMSNorm forward + backward**
   - Qwen3 uses RMSNorm everywhere (2 per layer × 28 layers + 1 final + 2 QK-Norm per layer = 85 instances)
   - HF implementation: `x.pow(2).mean(-1, keepdim=True)` → `rsqrt(var + eps)` → `weight * normalized`
   - Decomposition into existing ops works but is 5+ dispatches per norm — should be a fused Slang shader
   - Need: `rms_norm_fwd.slang` (single-pass: compute variance, normalize, scale) + `rms_norm_backward.slang`
   - Register as custom op (not aten schema — RMSNorm is not in ATen)

2. **nll_loss ignore_index support**
   - Current nll_loss shader indexes `input[target]` without checking `target == ignore_index`
   - Qwen3 training uses `ignore_index=-100` for padding tokens — current shader would index OOB or give wrong loss
   - Fix: Add `ignore_index` push constant to `nll_loss.slang`, skip samples where `target == ignore_index`

3. **nll_loss_backward on GPU**
   - No `nll_loss_backward` implemented — backward for cross_entropy currently falls through to CPU
   - Need: `nll_loss_backward.slang` shader + registration on PrivateUse1
   - Grad is `-1/N` at `input[n, target[n]]`, zero elsewhere (with ignore_index handling)

4. **cross_entropy backward on GPU**
   - Currently: `cross_entropy_loss` registered on PrivateUse1 blocks autograd decomposition
   - Options: (a) Remove PrivateUse1 registration and let PyTorch decompose (log_softmax + nll_loss), or (b) Register on AutogradPrivateUse1 with custom backward calling nll_loss_backward + log_softmax_backward
   - Must verify backward produces correct gradients through the full CE → log_softmax → nll_loss chain

5. **Large vocab embedding + linear (151936 classes)**
   - Vocab size 151936 with tied embeddings (`lm_head.weight = embed_tokens.weight`)
   - `embedding_dense_backward` must handle 151936-row gradient accumulation — verify no OOM or uint32 overflow
   - Final linear layer: `[B*S, 1024] @ [1024, 151936]` — verify `mm` handles wide output matrices

#### P1 — Required for Correct Training

6. **SwiGLU MLP**
   - Qwen3 FFN: `down_proj(silu(gate_proj(x)) * up_proj(x))` — 3 linear layers + SiLU + elementwise mul
   - Already have: `mm` (linear, no bias), `silu` + `silu_backward`, `mul`
   - Should work via decomposition, but verify backward through gated multiply: `d/dx[silu(gate(x)) * up(x)]`
   - Test: Write explicit Qwen3MLP test (forward + backward correctness)

7. **GQA head expansion (repeat_kv)**
   - `kv.unsqueeze(2).expand(B, KVH, n_rep, S, D).reshape(B, H, S, D)` where n_rep=2
   - Already have: `unsqueeze`, `expand`, `reshape`/`view`
   - Verify: 5D expand works, reshape from 5D→4D works, backward through expand/reshape is correct

8. **RoPE compatibility with Qwen3's apply_rotary_pos_emb**
   - Current: Custom `vulkan_rope(input, theta)` computes cos/sin internally from position index
   - Qwen3: Pre-computes `(cos, sin)` externally via `RotaryEmbedding`, then applies as `(q * cos) + (rotate_half(q) * sin)`
   - Issue: Qwen3's RoPE is applied via standard tensor ops (mul, cat, add), NOT our custom kernel
   - The decomposed version (mul, slice, cat, neg, mul, add) should work on GPU via existing ops
   - Verify: `rotate_half` uses slicing + neg + cat, all of which exist. Backward must flow through.

9. **Causal mask creation**
   - `torch.triu(torch.full((S, S), -inf), diagonal=1)` — already have `triu` + `full`
   - Verify: `full` with `-inf` value works on Vulkan, `triu` with diagonal offset works

10. **Dtype casting (float32 ↔ bfloat16)** — DONE
    - Widen-compute-narrow: bf16/f16 stored natively, cast to f32 for GPU compute, cast back
    - GPU cast shaders use uint32 bit manipulation (no VK_KHR_shader_float16_int8 needed)
    - All ops support f16/bf16 inputs via `ensure_float32()`/`cast_from_float32()` pattern

#### P2 — Performance (training will work but slowly without these)

11. **Fused RMSNorm shader**
    - Single-pass: read input, compute per-row variance, normalize, scale by weight
    - Avoids 5+ separate kernel launches from decomposed version
    - Critical: 85 RMSNorm instances per forward pass × 28 layers = most frequent op

12. **Fused SwiGLU shader**
    - Combine `silu(gate_proj(x)) * up_proj(x)` into single kernel to reduce memory traffic
    - Lower priority than RMSNorm since the three linears dominate compute

13. **Large-vocab softmax optimization**
    - log_softmax over 151936 classes — current per-row softmax shader may be slow for wide rows
    - Consider multi-pass softmax for rows > workgroup size (256)

14. **Gradient checkpointing support**
    - Qwen3 uses `GradientCheckpointingLayer` to trade compute for memory
    - Requires `torch.utils.checkpoint.checkpoint()` to work with Vulkan tensors
    - Verify: checkpoint recomputes forward during backward — all ops must be deterministic

#### P3 — Nice to Have

15. **Adam/AdamW fused kernel**
    - Currently uses `_foreach_*` ops (scalar loop over parameters)
    - Fused Adam shader would update param, momentum, variance in one dispatch per parameter

16. **Gradient clipping on GPU**
    - `torch.nn.utils.clip_grad_norm_()` — compute global norm, scale grads
    - Currently decomposes into norm + clamp — should work but verify

17. **BFloat16 native compute shaders**
    - Currently uses widen-compute-narrow (bf16→f32→bf16 per op), which works but adds cast overhead
    - Native bf16 shaders would avoid f32 upcasting — requires VK_KHR_shader_float16_int8 (SwiftShader may not support)

### Implementation Order

```
Phase 7A (Critical Path — enable training loop): DONE
  [x] nll_loss ignore_index fix (shader + C++)
  [x] nll_loss_backward (new shader + C++ + registration)
  [x] cross_entropy backward path (fix autograd chain)
  [x] RMSNorm forward + backward (fused Slang shaders)
  [x] Shape ops autograd wrappers (view, reshape, transpose, permute, etc.)
  [x] Test: Qwen3DecoderLayer forward + backward correctness
  [x] Test: Full Qwen3 training step (forward + CE + backward + SGD step)

Phase 7B (Correctness — verify all Qwen3 ops): DONE
  [x] SwiGLU MLP forward + backward test
  [x] GQA repeat_kv correctness test
  [x] RoPE via decomposed ops test
  [x] Causal mask test
  [x] cross_entropy with ignore_index backward test
  [x] End-to-end training step with ignore_index=-100
  [x] Tied embeddings test (tie_weights() after .vulkan())
  [x] Large-vocab (151936) verification (embedding, linear, softmax, cross_entropy, backward)

Phase 7C (Performance): DONE
  [x] Fused RMSNorm shader (single-pass with groupshared reduction)
  [x] Large-vocab softmax optimization (decompose into amax+exp+sum+div for rows > 256)
  [x] Fused SwiGLU shader (single-pass silu(gate)*up + backward, torch_vulkan.swiglu())
  [x] Gradient checkpointing verification (torch.utils.checkpoint works with Vulkan)
  [x] Gradient clipping verification (clip_grad_norm_ + clip_grad_value_)
```

---

## Implementation Progress

### Stages 1-4: IMPLEMENTED — all tests passing
### Stages 5-6: DONE — All ops on GPU, 206 Slang shaders, zero stubs
### Stage 7: NOT STARTED — deprioritized (needs multi-GPU hardware)
### Stage 8: DONE — torch.compile works with eager backend, all compile tests pass

### Phase 2-3: DONE — Generator, model coverage ops (triu/tril/pad/index/repeat/stack/erf/flip/roll/as_strided/resize_)
### Phase 4: DONE — Autocast fallthrough, GradScaler ops, AMP module APIs
### Phase 5: DONE — All ops on GPU, no CPU fallbacks. 206 Slang shaders compiled. Zero TORCH_CHECK stubs.
### Phase 7A-C: DONE — Qwen3 training verified. RMSNorm, nll_loss, cross_entropy backward, shape ops autograd, large-vocab softmax, fused SwiGLU, gradient checkpointing/clipping.
### VL Ops: masked_scatter (vision embedding injection), Conv3d (patch embedding via temporal decomposition), unbind/outer (decompose into existing ops)
### Image Gen: upsample_nearest2d_backward (UNet/diffusion backward), mini UNet training verified end-to-end
### FP8: float8_e4m3fn + float8_e5m2 via widen-compute-narrow, GPU cast shaders (4 values packed per uint32)
### Performance: Tiled matmul (shared memory), SDPA via bmm+softmax, GPU backward passes, fused RMSNorm/SwiGLU/softmax_backward/log_softmax_backward, deferred execution (cmd buffer batching), buffer quarantine (WAR hazard elimination), transpose-aware matmul, GPU strided copy, atomic embedding backward (zero preread flushes), movedim skip optimization, int32 buffer reinterpret for indices (saves 3 dispatches per CE step), smart barrier insertion (37.5% barrier skip rate on Qwen3 training step)
### Dependencies: Slang (third_party/slang submodule, v2025.8), VMA (third_party/VulkanMemoryAllocator)

### Performance Benchmarks (MiniQwen3 training step, RTX 4060 Ti, real GPU)

| Config | Phase | Dispatches | Time | vs CPU |
|--------|-------|-----------|------|--------|
| Large (4L, D=256, B=2, S=64) | Forward | 48 | 3.39ms | — |
| Large | CE+to_cpu | 2 | 0.23ms | — |
| Large | Backward | 106 | 6.84ms | — |
| Large | Optimizer | 6 | 0.50ms | — |
| **Large total** | — | **162** | **~11ms** | **~0.6x CPU** |

CPU baseline (equivalent PyTorch model): ~19.5ms/step.
Note: Vulkan faster than CPU for attention-heavy models after wave intrinsic flash attention (D=32: ~11ms vs ~19.5ms CPU). The wave-intrinsic path eliminates all barriers in attention forward/backward — the dominant cost for small D.
Dispatch count (Large, 4L): ~162 (fwd=48, bwd=106, opt=6 [batched SGD], causal mask internal to flash attention). Barrier skip rate: fwd=26%, bwd=61%, opt=100%.
Optimizations: deferred execution, buffer quarantine, transpose-aware matmul (mm_tiled + bmm shaders support transpose_a/transpose_b), GPU strided copy, buffer-specific pre-read flush, fused SGD shader, GPU nll_loss count, zero-copy view/reshape, contiguous-returns-self, removed F.linear monkey-patch, CPU scalar tensor shortcut (3→1 dispatch), same-shape binary op skip, same-shape expand skip, 4D causal mask pre-expansion, vulkan_mm_ex/vulkan_bmm_ex (avoid .t()/.transpose() GPU copy), stack-allocated dispatch buffers, cached runtime pointer, lock-free pipeline cache fast path, atomic embedding backward (zero preread flushes), fused log_softmax_backward (5→1 dispatch), movedim skip for last-dim ops, fused large-row softmax/log_softmax (7→1 fwd, 5→1 bwd for any row size), fused nll_loss_mean (CE: 7→4 dispatches), broadcast-aware add (suffix-broadcast [B,H,S,S]+[S,S] uses modulo shader, 1 dispatch instead of 2), true in-place add_/sub_/mul_/div_ + scalar variants (1 dispatch vs 2, saves ~62 dispatches per backward), fused layer_norm forward (8→1 dispatch), norm backward empty buffers (saves 2 dispatches per normalization backward), zero-copy N-D transpose + unsqueeze/squeeze direct-stride (metadata-only views, bmm_ex detects transposed layout), in-stride-aware expand shader (skip pre-contiguous for non-contiguous f32 inputs, saves 1 dispatch per GQA k/v expand), multi-tensor cat_n shader (N inputs in 1 dispatch instead of N-1, 9 bindings: in0..in7+out), scaled_bmm (scale*q@k.T fused into 1 dispatch, saves 1 per layer in attention), fused add_rms_norm forward (residual+shortcut+rms_norm in 1 dispatch vs 2, saves 7 dispatches/fwd with 3.5 fused calls/layer), fused add_rms_norm backward (rms_norm_bwd+add_grad in 1 pass, saves 7 dispatches/bwd), seq-major attention layout (auto-detect view+transpose → native [B,S,H,D] reads/writes in all 3 attention shaders, zero-copy output reshape, saves 16 fwd + 16 bwd dispatches in 4L model), inline D_i fusion (D_i computed inline in bwd+bwd_kv shaders, eliminates di dispatch, bwd: 3→2 dispatches, saves 4 bwd dispatches in 4L model), batched SGD (7 params/dispatch via 2D grid + config buffer, 39→6 optimizer dispatches for 39-param model), wave-intrinsic flash attention (D<=32: 32 threads + WaveActiveSum, zero barriers per key position — fwd -35%, bwd -52%, total -47% on RTX 4060 Ti at D=32), D<=64 flash attention fwd+bwd (64-thread d64 variants for all 3 attention shaders, 6 vs 8 barrier syncs per key position, benefits Qwen3-0.6B with head_dim=64).
- **f32 matmul skip chunking** (PERF): `vulkan_mm` and `vulkan_linear` skip CPU-staged K-chunking for f32 inputs since `ensure_float32` is a no-op and workgroup limits aren't exceeded. Fixes catastrophic 3-6s LM head forward for f32 models with `out_features * in_features > 16.7M` (e.g., V=32768, D=512).
- **2D cast shader dispatch** (FIX): `ensure_float32`/`cast_from_float32` now use 2D dispatch `(wg_x=65535, wg_y=ceil(...))` for tensors with >16.7M elements. Previously, large bf16/f16 tensors (Qwen3 LM head 151936×1024=155M) were silently casting only the first 16.7M elements (1D dispatch was clamped to 65535 workgroups). All 8 cast shaders updated to compute `idx = tid.y * 65535 * 256 + tid.x`.
Remaining gap: backward pass — per-dispatch Vulkan API overhead (~0.03-0.07ms × 175 dispatches). Kernel fusion via torch.compile custom backend would be next major optimization.
