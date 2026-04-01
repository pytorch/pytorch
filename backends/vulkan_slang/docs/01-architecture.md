# Architecture & Project Structure

## Project Overview

Build a complete, out-of-tree PyTorch backend that uses the Vulkan compute API to execute tensor operations on any Vulkan-capable GPU. Unlike the existing PyTorch Vulkan backend (inference-only, now deprecated in favor of ExecuTorch), this project targets **full training support**: forward pass, autograd/backward pass, optimizer steps, mixed precision, and eventually distributed training.

The backend registers via PyTorch's `PrivateUse1` dispatch key mechanism (renamed to `vulkan`) and ships as a standalone Python package (`torch_vulkan`) that auto-loads when `import torch` is called.

### Target User Experience

```python
import torch
import torch_vulkan  # auto-registers via entry_points

device = torch.device("vulkan:0")
model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    x, y = batch[0].to(device), batch[1].to(device)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Why PrivateUse1

PyTorch's recommended path for out-of-tree accelerators since PyTorch 2.1. Provides dispatch keys `PrivateUse1` and `AutogradPrivateUse1`, device guard integration, generator support, storage hooks, AMP support, and the ability to rename the device to a custom string (e.g., `"vulkan"`). No need to fork PyTorch. Reference: https://docs.pytorch.org/tutorials/advanced/privateuseone.html

---

## Directory Structure

```
torch_vulkan/                          # Python package root
├── setup.py                           # Build config with CMake extension
├── CMakeLists.txt                     # Top-level CMake
├── csrc/                              # C++ source
│   ├── vulkan/                        # Vulkan runtime abstraction
│   │   ├── Context.h/.cpp             # VkInstance, VkDevice, queue management
│   │   ├── Device.h/.cpp              # Physical/logical device wrapper
│   │   ├── CommandBuffer.h/.cpp       # Command buffer pool & recording
│   │   ├── Memory.h/.cpp              # VMA-based allocator, buffer/image management
│   │   ├── Pipeline.h/.cpp            # Compute pipeline cache & creation
│   │   ├── ShaderModule.h/.cpp        # SPIR-V loading
│   │   ├── Sync.h/.cpp                # Fences, semaphores, events, timeline semaphores
│   │   ├── Stream.h/.cpp              # Stream abstraction (maps to VkQueue + cmd buffers)
│   │   └── DescriptorSet.h/.cpp       # Descriptor pool/set management
│   ├── ops/                           # ATen operator implementations
│   │   ├── TensorFactories.cpp        # empty, zeros, ones, rand, etc.
│   │   ├── UnaryOps.cpp               # neg, abs, exp, log, sin, cos, sqrt, rsqrt, etc.
│   │   ├── BinaryOps.cpp              # add, sub, mul, div, pow, etc.
│   │   ├── ReduceOps.cpp              # sum, mean, max, min, argmax, etc.
│   │   ├── BlasOps.cpp                # mm, bmm, addmm, linear
│   │   ├── ConvOps.cpp                # conv1d, conv2d, conv_transpose2d
│   │   ├── PoolOps.cpp                # max_pool2d, avg_pool2d, adaptive_avg_pool2d
│   │   ├── NormOps.cpp                # batch_norm, layer_norm, group_norm
│   │   ├── ActivationOps.cpp          # relu, gelu, silu, sigmoid, tanh, softmax
│   │   ├── LossOps.cpp                # cross_entropy, mse_loss, nll_loss
│   │   ├── IndexOps.cpp               # index_select, gather, scatter, index_put
│   │   ├── ShapeOps.cpp               # view, reshape, permute, transpose, cat, split
│   │   ├── CopyOps.cpp                # copy_, to, clone, contiguous
│   │   ├── CompareOps.cpp             # eq, ne, lt, gt, le, ge, where
│   │   ├── EmbeddingOps.cpp           # embedding, embedding_backward
│   │   ├── AttentionOps.cpp           # scaled_dot_product_attention (flash-attn style)
│   │   ├── RandomOps.cpp              # uniform_, normal_, bernoulli_, dropout
│   │   └── FallbackOps.cpp            # CPU fallback for unimplemented ops
│   ├── backend/                       # PyTorch backend integration
│   │   ├── Registration.cpp           # TORCH_LIBRARY_IMPL for PrivateUse1
│   │   ├── AutogradRegistration.cpp   # TORCH_LIBRARY_IMPL for AutogradPrivateUse1
│   │   ├── DeviceGuard.cpp            # c10::impl::DeviceGuardImpl
│   │   ├── Allocator.cpp              # at::Allocator for Vulkan memory
│   │   ├── Generator.cpp              # at::GeneratorImpl (Philox on GPU)
│   │   ├── Hooks.cpp                  # at::PrivateUse1HooksInterface
│   │   ├── Serialization.cpp          # Storage serialization for save/load
│   │   └── Event.cpp                  # at::Event for synchronization
│   ├── autocast/                      # Automatic mixed precision
│   │   └── AutocastRegistration.cpp   # FP16/BF16 autocast policies
│   └── init.cpp                       # Python module initialization
├── shaders/                           # All Slang shaders (see 03-slang-shaders.md)
├── tools/
│   ├── compile_shaders.py             # slangc → SPIR-V (+ backward entries) + C++ embedding
│   ├── compile_cpu_tests.py           # slangc → C++ for CPU shader unit tests
│   ├── generate_op_list.py            # Generate op coverage report
│   └── slang_version.txt             # Pinned Slang compiler version
├── cpu_tests/                         # CPU-side shader math tests (no Vulkan needed)
│   ├── test_unary_cpu.cpp
│   ├── test_binary_cpu.cpp
│   ├── test_activation_cpu.cpp
│   ├── test_matmul_cpu.cpp
│   ├── test_autodiff_cpu.cpp
│   └── CMakeLists.txt
├── python/
│   └── torch_vulkan/
│       ├── __init__.py                # Auto-registration entry point
│       ├── _C.pyi                     # Type stubs for C++ extension
│       ├── amp.py                     # AMP GradScaler integration
│       ├── profiler.py                # Profiler hooks
│       └── testing.py                 # Test utilities
├── tests/                             # Full integration tests (SwiftShader OK)
│   ├── test_basic_ops.py
│   ├── test_matmul.py
│   ├── test_conv.py
│   ├── test_autograd.py
│   ├── test_training.py
│   ├── test_amp.py
│   ├── test_serialization.py
│   ├── test_slang_autodiff.py
│   ├── conftest.py
│   └── benchmarks/                    # Require real GPU
│       ├── bench_matmul.py
│       ├── bench_conv.py
│       └── bench_e2e.py
└── docs/
    ├── ARCHITECTURE.md
    ├── ADDING_OPS.md
    ├── SHADER_GUIDE.md
    ├── CPU_DEVELOPMENT.md
    └── BENCHMARKING.md
```

---

## Key Technical Decisions

### Memory Layout
Row-major contiguous by default. Non-contiguous tensors use stride-aware indexing. Raw byte buffers (`VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`), not images.

### Synchronization Model
Stream-ordered, matching CUDA semantics. Default stream = sequential. Events for cross-stream sync. `synchronize()` = block until idle.

### Operator Priority

1. **P0 (Must-have):** `copy_`, `add`, `mul`, `mm`, `bmm`, `addmm`, `convolution`, `batch_norm`, `layer_norm`, `relu`, `softmax`, `cross_entropy`, `empty`, `zeros`, `ones`, `cat`, `view`, `reshape`, `permute`, `transpose`, `contiguous`, `to`, `sum`, `mean`
2. **P1 (Core training):** `embedding`, `dropout`, `gelu`, `silu`, `linear`, `adaptive_avg_pool2d`, `max_pool2d`, `index_select`, `gather`, `scatter`, `log_softmax`, `nll_loss`, `fill_`, `zero_`, `clone`, `expand`, `unsqueeze`, `squeeze`
3. **P2 (Advanced models):** `scaled_dot_product_attention`, `group_norm`, `interpolate`, `grid_sample`, `conv_transpose2d`, `topk`, `sort`, `cumsum`
4. **P3 (Optimization):** Fused optimizer kernels, RoPE, remaining aten ops

---

## Dependencies & Toolchain

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Vulkan SDK | ≥ 1.3.296 | Headers, validation layers, bundled slangc |
| Slang compiler | latest stable | `.slang` → SPIR-V and `.slang` → C++ |
| SwiftShader | latest | CPU-based Vulkan for dev/CI (no GPU needed) |
| VMA | latest | Vulkan Memory Allocator |
| PyTorch | ≥ 2.1 | PrivateUse1 backend support |
| Python | ≥ 3.9 | Package + tests |
| pytest | latest | Test runner |
| CMake | ≥ 3.18 | Build system |

### Building (no GPU required)

```bash
git clone https://github.com/user/torch-vulkan.git && cd torch-vulkan
git submodule update --init  # VMA

# Run CPU shader tests (no Vulkan needed at all)
cd cpu_tests && mkdir build && cd build && cmake .. && make && ctest

# Build full package and run integration tests on SwiftShader
export VK_ICD_FILENAMES=/usr/share/swiftshader/vk_swiftshader_icd.json
pip install -e .
pytest tests/ -x --timeout=300
```

---

## Coding Standards

### Slang Shaders
- Target SPIR-V for Vulkan 1.2
- `[Differentiable]` on all functions needing backward passes
- Generics (`<T : IFloat>`) for dtype variants — never preprocessor defines
- `import` for code reuse, one file per op or closely related op family
- Entry points: `computeMain` (forward), `bwd_computeMain` (backward)
- `groupshared` for shared memory, `GroupMemoryBarrierWithGroupSync()` for barriers
- Default: `[numthreads(256, 1, 1)]` for 1D, `[numthreads(16, 16, 1)]` for 2D

### C++
- C++17, `TORCH_CHECK` / `TORCH_INTERNAL_ASSERT`, RAII for Vulkan objects
- Shader-language-agnostic: loads SPIR-V byte arrays, doesn't know they came from Slang

### Python
- Type hints, docstrings, `torch.testing.assert_close()`

### Git
- Feature branches per stage, one commit per operator
- All tests pass on SwiftShader before merge
