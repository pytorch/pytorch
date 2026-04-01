# PyTorch Backend Integration

This document covers registering the Vulkan backend with PyTorch's `PrivateUse1` mechanism and implementing the autograd backward pass.

---

## Device Registration (`csrc/backend/`)

### Core Registration

- [ ] `c10::register_privateuse1_backend("vulkan")` — renames PrivateUse1 to "vulkan"
- [ ] Python entry point in `setup.py` so `import torch; import torch_vulkan` auto-registers

### DeviceGuard (`DeviceGuard.cpp`)

Implements `c10::impl::DeviceGuardImpl<PrivateUse1>`:

- [ ] `setDevice(DeviceIndex)` — switch active Vulkan device
- [ ] `getDevice()` — return current device index
- [ ] `deviceCount()` — number of Vulkan devices (SwiftShader = 1)
- [ ] `exchangeDevice(DeviceIndex)` — atomic swap
- [ ] `getStream(DeviceIndex)` — return current stream for device

**Testing gate:**
- [ ] `torch.vulkan.device_count()` returns ≥1 on SwiftShader
- [ ] `torch.vulkan.is_available()` returns True on SwiftShader
- [ ] Device guard context manager works: `with torch.vulkan.device(0): ...`

### Allocator (`Allocator.cpp`)

Implements `at::Allocator` backed by VMA:

- [ ] `allocate(size)` — allocate VulkanBuffer via VMA
- [ ] `deallocate(ptr)` — free through VMA
- [ ] Register as default allocator for PrivateUse1 tensors

**Testing gate:**
- [ ] `torch.empty(4, 4, device="vulkan:0")` allocates without error
- [ ] Tensor destruction frees memory (no VMA leak warnings)
- [ ] Large allocation + free cycle (1000 iterations) doesn't grow memory

### Generator (`Generator.cpp`)

Implements `at::GeneratorImpl` for Vulkan:

- [ ] Philox4x32-10 state management
- [ ] Seed get/set
- [ ] State serialization for reproducibility
- [ ] Integration with `torch.manual_seed()`

**Testing gate:**
- [ ] `torch.manual_seed(42)` on Vulkan produces deterministic output from `torch.randn`
- [ ] Two generators with same seed produce identical sequences

### Hooks (`Hooks.cpp`)

Implements `at::PrivateUse1HooksInterface`:

- [ ] `hasPrimaryContext(device_index)` 
- [ ] `getDefaultGenerator(device_index)`
- [ ] `isPinnedPtr(data)` — for pinned memory support
- [ ] `resizePrivateUse1Bytes(storage, size)`

### Serialization (`Serialization.cpp`)

- [ ] `torch.save()` — Vulkan tensors serialize to CPU then save
- [ ] `torch.load(map_location="vulkan:0")` — load to CPU, then copy to Vulkan
- [ ] Storage registration for custom serialization

**Testing gate:**
- [ ] Round-trip: create tensor on Vulkan → `torch.save()` → `torch.load()` on Vulkan → values match
- [ ] `map_location` works correctly

### Event (`Event.cpp`)

- [ ] `at::Event` wrapping Vulkan fence/timeline semaphore
- [ ] `record()`, `wait()`, `query()`, `elapsed_time()`

**Testing gate:**
- [ ] Event recorded on stream, waited on host, completes correctly

---

## Dispatch Key Registration

### Forward Ops (`Registration.cpp`)

```cpp
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", vulkan_add);
    m.impl("mm", vulkan_mm);
    m.impl("relu", vulkan_relu);
    m.impl("convolution_overrideable", vulkan_conv);
    // ... all forward ops
}
```

- [ ] Register all P0 ops (see `01-architecture.md` for priority list)
- [ ] Register all P1 ops
- [ ] Fallback to CPU for unregistered ops (`FallbackOps.cpp`)

**Testing gate:**
- [ ] `torch.add(a.vulkan(), b.vulkan())` dispatches to Vulkan kernel (not CPU fallback)
- [ ] Unregistered op falls back to CPU with warning, produces correct result
- [ ] `torch.ops.aten.add.Tensor` resolves to Vulkan implementation

### Autograd Registration (`AutogradRegistration.cpp`)

```cpp
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("scaled_dot_product_attention", vulkan_sdpa_autograd);
}
```

- [ ] Register Tier 3 backward ops as `torch::autograd::Function` subclasses
- [ ] Flash attention forward+backward as custom autograd function
- [ ] All Tier 1 ops rely on Slang autodiff (backward SPIR-V registered at PrivateUse1 level)
- [ ] All Tier 2 ops rely on PyTorch decomposition (no registration needed)

**Testing gate:**
- [ ] `torch.autograd.gradcheck()` passes for every differentiable op
- [ ] `loss.backward()` on a simple MLP produces correct gradients on SwiftShader
- [ ] Custom autograd function for flash attention: forward + backward correct

---

## Autocast / Mixed Precision (`autocast/AutocastRegistration.cpp`)

- [ ] Register autocast policies on `AutocastPrivateUse1` dispatch key
- [ ] Policies: GEMM/conv → f16, norms/losses → f32, element-wise → preserve
- [ ] `torch.amp.autocast("vulkan")` context manager works
- [ ] Integration with `torch.amp.GradScaler` (via `python/torch_vulkan/amp.py`)

**Testing gate:**
- [ ] Autocast promotes matmul inputs to f16, keeps loss in f32
- [ ] GradScaler inf-check and unscale work on Vulkan tensors
- [ ] AMP training loop converges on MNIST (SwiftShader)

---

## Python Module (`python/torch_vulkan/`)

### `__init__.py`
- [ ] Auto-register backend via entry points (no explicit call needed)
- [ ] Expose `torch.vulkan.is_available()`, `device_count()`, `get_device_name()`
- [ ] `synchronize()` helper

### `amp.py`
- [ ] GradScaler subclass for Vulkan-specific AMP quirks

### `testing.py`
- [ ] `skip_if_no_vulkan` decorator
- [ ] `vulkan_device` fixture
- [ ] Standard tolerance presets (SwiftShader may be slightly less precise)

### `profiler.py`
- [ ] PyTorch profiler hooks for Vulkan dispatch timing

**Testing gate:**
- [ ] `import torch_vulkan` succeeds and registers backend
- [ ] `torch.device("vulkan:0")` works after import
- [ ] All public API functions return sane values

---

## Reference Materials

- PrivateUse1 Tutorial: https://docs.pytorch.org/tutorials/advanced/privateuseone.html
- Extending Dispatcher: https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html
- Accelerator Integration: https://docs.pytorch.org/docs/stable/accelerator/index.html
- OpenReg Example: https://docs.pytorch.org/docs/main/accelerator/hooks.html
- PyTorch MPS Backend (closest analogue): `pytorch/pytorch/aten/src/ATen/mps/`
- ExecuTorch Vulkan: https://github.com/pytorch/executorch/tree/main/backends/vulkan
