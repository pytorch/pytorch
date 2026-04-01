# Vulkan Runtime Layer

The Vulkan runtime (`csrc/vulkan/`) provides a thin, RAII-based abstraction over the Vulkan compute API. It is completely independent of PyTorch — it only knows about Vulkan objects and SPIR-V byte arrays.

---

## Components

### Context (`Context.h/.cpp`)

Global singleton managing the Vulkan instance and device lifecycle.

- [ ] `VkInstance` creation with validation layers (always enabled in debug/SwiftShader)
- [ ] Physical device enumeration (SwiftShader shows as a device)
- [ ] `VkDevice` creation with compute queue families
- [ ] Feature detection: `float16`, `int8`, `subgroupSize`, `KHR_cooperative_matrix`
- [ ] Multi-device support (device index → VkDevice mapping)
- [ ] Debug messenger callback for validation layer output

**Testing gate:**
- [ ] Unit test: Context boots on SwiftShader, enumerates at least one device
- [ ] Unit test: Feature detection returns sane values on SwiftShader (no float64, subgroup may be 1)
- [ ] Unit test: Multi-device index out-of-range throws clean error

### Device (`Device.h/.cpp`)

Physical + logical device wrapper exposing capabilities and queue families.

- [ ] Physical device properties and limits query
- [ ] Queue family selection (compute-capable)
- [ ] Device extension negotiation
- [ ] Device name and type reporting (for `torch.vulkan.get_device_name()`)

**Testing gate:**
- [ ] Unit test: Device wrapper exposes correct name string on SwiftShader

### Memory (`Memory.h/.cpp`)

VMA-based GPU memory management.

- [ ] VMA integration for suballocations
- [ ] Buffer types: device-local, host-visible-coherent, staging
- [ ] `VulkanBuffer` RAII class (constructor allocates, destructor frees)
- [ ] Memory pool with slab allocator for small allocations
- [ ] Buffer-to-buffer copy via staging buffers
- [ ] Host↔device transfer utilities

**Testing gate:**
- [ ] Unit test: Allocate device-local buffer, write via staging, read back, verify contents
- [ ] Unit test: RAII destructor properly frees — allocate/free 10K buffers without leak
- [ ] Unit test: OOM on SwiftShader produces a clean error (not a crash)

### CommandBuffer (`CommandBuffer.h/.cpp`)

Command buffer pool and recording abstraction.

- [ ] Command buffer pool per-thread (avoids locking)
- [ ] `begin()` / `end()` / `submit()` lifecycle
- [ ] Pipeline barrier insertion helpers
- [ ] One-shot command buffer utility for simple transfers

**Testing gate:**
- [ ] Unit test: Record and submit a no-op command buffer on SwiftShader
- [ ] Unit test: Multiple command buffers from same pool don't interfere

### Stream (`Stream.h/.cpp`)

Stream abstraction matching CUDA stream semantics.

- [ ] `Stream` class: VkQueue + rotating command buffer pool
- [ ] Default stream = sequential execution
- [ ] Async dispatch with fence-based completion
- [ ] `synchronize()` blocks until queue idle
- [ ] Stream query (is idle?)

**Testing gate:**
- [ ] Unit test: Default stream serializes two dispatches correctly
- [ ] Unit test: `synchronize()` returns only after work completes
- [ ] Unit test: Dispatch + read-back produces correct result

### Pipeline (`Pipeline.h/.cpp`)

Compute pipeline management.

- [ ] Load SPIR-V from embedded byte arrays
- [ ] `VkShaderModule` creation from SPIR-V
- [ ] `VkComputePipeline` creation with specialization constants
- [ ] Pipeline cache with disk serialization (`VkPipelineCache`)
- [ ] Thread-safe pipeline lookup (hash on SPIR-V + spec constants)

**Testing gate:**
- [ ] Unit test: Load trivial SPIR-V, create pipeline, dispatch on SwiftShader
- [ ] Unit test: Pipeline cache hit on second creation of same shader
- [ ] Unit test: Invalid SPIR-V produces clean error

### DescriptorSet (`DescriptorSet.h/.cpp`)

Descriptor pool and set management for binding buffers to shaders.

- [ ] Descriptor pool with configurable max sets
- [ ] Descriptor set layout creation from reflection or manual spec
- [ ] Bind storage buffers to descriptor set slots
- [ ] Pool exhaustion → automatic pool growth
- [ ] Per-frame descriptor set recycling

**Testing gate:**
- [ ] Unit test: Bind two buffers to a descriptor set, dispatch shader that reads both
- [ ] Unit test: Pool exhaustion triggers growth (not crash)

### Sync (`Sync.h/.cpp`)

Synchronization primitives.

- [ ] Fence wrappers (create, wait, reset)
- [ ] Semaphore wrappers (for cross-queue sync if needed)
- [ ] Event wrappers (for PyTorch `at::Event` integration)
- [ ] Timeline semaphores (Vulkan 1.2)

**Testing gate:**
- [ ] Unit test: Fence signals after dispatch completes
- [ ] Unit test: Wait on unsignaled fence times out correctly

---

## Integration Test: End-to-End Trivial Compute Shader

After all runtime components are implemented, the following must work on SwiftShader:

```
1. Context boots, selects SwiftShader device
2. Allocate two buffers (A filled with 1.0, B filled with 2.0)
3. Load trivial add.spv shader
4. Create pipeline, bind descriptors, dispatch
5. Read back result buffer → every element is 3.0
6. All Vulkan objects cleaned up without validation errors
```

- [ ] **Integration test passes on SwiftShader**
- [ ] **No validation layer errors or warnings**
- [ ] **Memory leak check (all VMA allocations freed)**

---

## Design Principles

1. **RAII everywhere.** Every Vulkan handle wrapped in a class whose destructor calls `vkDestroy*`. No manual cleanup paths.
2. **Validation layers always on in debug.** SwiftShader is the primary dev target — validation catches API misuse early.
3. **Thread safety.** Command buffer pools are per-thread. Pipeline cache is locked. Context is singleton with thread-safe init.
4. **No shader language awareness.** The runtime loads SPIR-V byte arrays. It doesn't know or care that they came from Slang. This keeps the runtime reusable.
