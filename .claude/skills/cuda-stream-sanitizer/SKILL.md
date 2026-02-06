---
name: cuda-stream-sanitizer
description: Debug CUDA stream synchronization issues and data races using CSAN (CUDA Sanitizer). Use when investigating race conditions between CUDA streams, debugging incorrect tensor access across streams, or when you see non-deterministic CUDA results. Also use when the user mentions TORCH_CUDA_SANITIZER, stream synchronization, or concurrent CUDA kernel issues.
---

# CUDA Stream Sanitizer (CSAN)

Debug CUDA stream synchronization issues and data races in PyTorch programs.

## When to Use

Use CSAN when you observe:
- Non-deterministic results on GPU
- Suspected race conditions between CUDA streams
- Incorrect tensor values when using multiple streams
- Debugging concurrent kernel execution issues

## Quick Start

Enable the sanitizer by setting the environment variable:

```bash
TORCH_CUDA_SANITIZER=1 python your_script.py
```

Or enable it programmatically:

```python
from torch.cuda._sanitizer import enable_cuda_sanitizer
enable_cuda_sanitizer()
```

## Understanding the Output

When CSAN detects a data race, it provides detailed information:

```
============================
CSAN detected a possible data race on tensor with data pointer 139719969079296
Access by stream 94646435460352 during kernel:
aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
writing to argument(s) self, out, and to the output
With stack trace:
  File "example.py", line 6, in <module>
    torch.mul(a, 5, out=a)
  ...

Previous access by stream 0 during kernel:
aten::rand(int[] size, *, int? dtype=None, Device? device=None) -> Tensor
writing to the output
With stack trace:
  File "example.py", line 3, in <module>
    a = torch.rand(10000, device="cuda")
  ...
```

The output tells you:
1. Which tensor has a race condition (by data pointer)
2. Which streams are conflicting
3. Which operations are accessing the tensor
4. Full stack traces for both accesses
5. Which arguments correspond to the affected tensor

## Common Issues and Fixes

### Issue: Tensor modified on different stream without synchronization

**Bad code:**
```python
a = torch.rand(4, 2, device="cuda")

with torch.cuda.stream(torch.cuda.Stream()):
    torch.mul(a, 5, out=a)  # Race condition!
```

**Fix: Wait for the default stream:**
```python
a = torch.rand(4, 2, device="cuda")

with torch.cuda.stream(torch.cuda.Stream()) as stream:
    stream.wait_stream(torch.cuda.default_stream())
    torch.mul(a, 5, out=a)  # Safe
```

### Issue: Producer-consumer pattern without sync

**Bad code:**
```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    result = compute_something(x)

with torch.cuda.stream(stream2):
    use_result(result)  # Race condition!
```

**Fix: Use events for synchronization:**
```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    result = compute_something(x)
    event = torch.cuda.Event()
    event.record()

with torch.cuda.stream(stream2):
    event.wait()  # Wait for stream1 to complete
    use_result(result)  # Safe
```

### Issue: Forgetting to sync before CPU access

**Bad code:**
```python
with torch.cuda.stream(torch.cuda.Stream()):
    result = model(x)

print(result.cpu())  # May read incomplete data!
```

**Fix: Synchronize before CPU transfer:**
```python
with torch.cuda.stream(torch.cuda.Stream()) as stream:
    result = model(x)

stream.synchronize()  # Or torch.cuda.synchronize()
print(result.cpu())  # Safe
```

## API Reference

```python
from torch.cuda._sanitizer import enable_cuda_sanitizer

# Enable CSAN (must be called before CUDA operations)
enable_cuda_sanitizer()
```

## Limitations

- CSAN is a prototype feature and may have false positives
- Adds overhead to CUDA operations
- Must be enabled before any CUDA operations occur
- Does not detect all types of race conditions

## Best Practices for Stream Safety

1. **Always synchronize when crossing stream boundaries**
   - Use `stream.wait_stream()` or `stream.wait_event()`

2. **Use events for fine-grained synchronization**
   - `event = torch.cuda.Event(); event.record(); event.wait()`

3. **Synchronize before CPU access**
   - Call `torch.cuda.synchronize()` or `stream.synchronize()`

4. **Consider using the default stream for simple cases**
   - The default stream has implicit synchronization with all streams

5. **Document stream assumptions in your code**
   - Make it clear which stream tensors are expected to be on
