# How `.backward()` Knows to Use the Compiled Backward Pass

## The Key Mechanism: Autograd Graph Connection

When you call `loss.backward()`, PyTorch **doesn't know** that the loss was computed with a compiled forward. Instead, it uses the **autograd graph** that was built during the forward pass.

## How It Works

### 1. **Forward Pass Creates Autograd Nodes**

When you run the compiled forward:

```python
compiled_model = torch.compile(model)
output = compiled_model(input_data)  # This creates a CompiledFunction in the graph
loss = criterion(output, target_data)
```

The key class is `CompiledFunction(torch.autograd.Function)` in `/home/lsakka/pytorch10/pytorch/torch/_functorch/_aot_autograd/runtime_wrappers.py:2103`:

```python
class CompiledFunction(torch.autograd.Function):
    compiled_fw = compiled_fw_func
    compiled_bw = compiled_bw_func
    
    @staticmethod
    def forward(ctx, *deduped_flat_tensor_args):
        # Runs the compiled forward graph
        fw_outs = call_func_at_runtime_with_args(
            CompiledFunction.compiled_fw,
            args,
            disable_amp=disable_amp,
        )
        # ... saves tensors for backward ...
        return tuple(raw_returns)
    
    @staticmethod
    def backward(ctx, *flat_args):
        # This gets called automatically by loss.backward()
        out = call_func_at_runtime_with_args(
            CompiledFunction.compiled_bw,
            all_args,
            steal_args=True,
            disable_amp=disable_amp,
        )
        return out
```

### 2. **Tensor's `grad_fn` Points to CompiledFunction**

When `CompiledFunction.forward()` returns tensors, those tensors have their `grad_fn` attribute set to `CompiledFunctionBackward`:

```python
output = compiled_model(input_data)
print(output.grad_fn)  # <CompiledFunctionBackward object>
```

This is **automatic** - PyTorch's autograd system automatically creates this connection when you subclass `torch.autograd.Function`.

### 3. **`loss.backward()` Walks the Autograd Graph**

When you call `loss.backward()`:

```python
loss = criterion(output, target_data)
loss.backward()  # Triggers the autograd engine
```

PyTorch's autograd engine:
1. Starts from the `loss` tensor
2. Walks backward through the computation graph
3. For each node, calls its `.backward()` method
4. When it reaches `CompiledFunctionBackward`, it calls `CompiledFunction.backward()`
5. `CompiledFunction.backward()` calls the **compiled backward graph**

## Visual Flow

```
User Code:
    output = compiled_model(input)
       ↓
    Creates tensors with grad_fn = CompiledFunctionBackward
       ↓
    loss = criterion(output, target)
       ↓
    loss.backward()  ← User calls this
       ↓
Autograd Engine:
    Walks graph → Finds CompiledFunctionBackward
       ↓
    Calls CompiledFunction.backward()
       ↓
    Executes compiled_bw(saved_tensors, grad_outputs)
       ↓
    Returns gradients tuple (grad_W, grad_b, ...)
       ↓
Autograd Engine:
    Accumulates gradients into .grad attributes
```

## Key Insight

**`.backward()` is NOT compiled**. What gets compiled is:
- ✅ The **forward computation graph** → `compiled_fw`
- ✅ The **backward computation graph** → `compiled_bw`

But the **autograd mechanism itself** (walking the graph, calling backward functions) is **NOT compiled** - it's the normal PyTorch autograd engine.

The compiled backward is just **one node** in the larger autograd graph. If you have:

```python
output1 = compiled_model(input)     # Compiled forward
output2 = some_other_op(output1)    # Regular eager op
loss = output2.sum()
loss.backward()
```

The autograd graph will be:
```
loss → SumBackward → some_other_op → CompiledFunctionBackward → ...
       [eager]       [eager]          [COMPILED]
```

## When is the Backward Compiled?

The backward graph can be compiled in two modes:

### 1. **Eager Compilation (Default for AOT Autograd)**
- Backward is compiled during the first forward pass
- Located at line 1909 in `runtime_wrappers.py`

### 2. **Lazy Compilation (Used by torch.compile)**
- Backward is compiled the first time `.backward()` is called
- Located at line 2468 in `runtime_wrappers.py`

```python
if CompiledFunction.compiled_bw is None:
    # First backward call - compile the backward graph now
    CompiledFunction.compiled_bw = aot_config.bw_compiler(
        bw_module, placeholder_list
    )
```

## Summary

**Q: How does `loss.backward()` know to use the compiled backward?**

**A:** It doesn't need to know! The compiled forward creates output tensors whose `grad_fn` points to `CompiledFunction.backward()`. When you call `loss.backward()`, PyTorch's autograd engine walks the graph and automatically calls `CompiledFunction.backward()`, which internally calls the compiled backward graph.

The magic is in the **autograd graph connection**, not in `.backward()` itself!
