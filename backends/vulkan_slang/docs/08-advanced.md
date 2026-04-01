# Advanced Features: Distributed Training & torch.compile

Covers Stages 7 (distributed) and 8 (torch.compile integration).

---

## Stage 7: Distributed Training & Multi-GPU

**Goal:** Multi-GPU training with DDP.

### Multi-GPU Foundation

- [ ] Multi-GPU via separate VkDevice per physical GPU
- [ ] Device-to-device copy (peer transfer via staging if no peer access)
- [ ] Per-device memory pools and command queues
- [ ] `torch.vulkan.device_count()` returns actual GPU count

**Testing gate:**
- [ ] On a 2-GPU system: allocate tensor on each device, copy between them
- [ ] Each device has independent stream and command buffer pool

### ProcessGroup Backend

- [ ] Initial approach: Vulkan→CPU→Gloo→CPU→Vulkan (functional, not fast)
- [ ] `all_reduce`, `broadcast`, `all_gather`, `reduce_scatter`
- [ ] Register as ProcessGroup backend for Vulkan device

**Testing gate:**
- [ ] `all_reduce` on 2 processes with Vulkan tensors produces correct result
- [ ] `broadcast` from rank 0 to all ranks works

### DistributedDataParallel

- [ ] `torch.nn.parallel.DistributedDataParallel` works with Vulkan model
- [ ] Gradient synchronization via the ProcessGroup backend
- [ ] Bucket-based gradient reduction

**Testing gate:**
- [ ] DDP training on 2 GPUs: loss matches single-GPU training (statistically)
- [ ] DDP training converges on CIFAR-10

### Future: Direct GPU Communication

- [ ] Vulkan peer-to-peer memory (if supported by hardware)
- [ ] Custom all-reduce using Vulkan compute shaders
- [ ] PCIe/NVLink-aware topology optimization

---

## Stage 8: torch.compile Integration

**Goal:** `torch.compile()` works with Vulkan backend.

### FakeTensor (Meta) Kernels

- [ ] Register meta implementations for all ops
- [ ] Meta kernels compute output shape/dtype without running actual compute
- [ ] Required for torch.compile's symbolic tracing

**Testing gate:**
- [ ] `torch.compile(model, backend="eager")` traces successfully with Vulkan tensors
- [ ] Output shapes match eager-mode execution

### AOT Autograd Compatibility

- [ ] Verify AOT Autograd can trace through Vulkan ops
- [ ] Backward graph construction works with Vulkan autograd functions
- [ ] Handle ops that AOT Autograd decomposes vs ops that stay as-is

**Testing gate:**
- [ ] `torch.compile(model)` with default backend runs inference correctly
- [ ] `torch.compile(model)` with training mode: forward + backward correct

### (Stretch) Inductor Backend for Vulkan

- [ ] Custom Inductor backend that generates Slang → SPIR-V from ATen-IR
- [ ] Kernel fusion at the Inductor level (fuse element-wise chains)
- [ ] Leverages Slang's modular compilation for generated kernels

**Testing gate:**
- [ ] Inductor-compiled model matches eager-mode output
- [ ] Fused kernels show fewer dispatches than eager mode
- [ ] Performance improvement over eager Vulkan execution

---

## Maturity Milestones

| Milestone | Stage | Verification |
|-----------|-------|-------------|
| First tensor on Vulkan | 1 | `torch.zeros(4,4, device="vulkan:0").cpu()` works |
| MLP inference | 2 | 3-layer MLP forward pass matches CPU |
| ResNet-18 inference | 2 | Full model forward pass matches CPU |
| GPT-2 inference | 2 | 124M model forward pass matches CPU |
| MNIST training | 3 | Trains to >95% accuracy on SwiftShader |
| CIFAR-10 training | 4 | Trains with Adam, lr scheduler |
| AMP training | 5 | f16 matmul, f32 norms, GradScaler |
| ResNet-50 AMP | 5 | Competitive throughput on real GPU |
| DDP training | 7 | Multi-GPU gradient sync |
| torch.compile | 8 | Compiled model matches eager |
