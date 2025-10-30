# Understanding PyTorch Autograd and grad_fn

## What is `grad_fn`?

`grad_fn` is a pointer to the **backward function** that created a tensor. It's PyTorch's way of building a **computation graph** for automatic differentiation.

## Simple Example

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y + 3

print(f"x.grad_fn: {x.grad_fn}")  # None (leaf node)
print(f"y.grad_fn: {y.grad_fn}")  # <PowBackward0>
print(f"z.grad_fn: {z.grad_fn}")  # <AddBackward0>
```

Output:
```
x.grad_fn: None
y.grad_fn: <PowBackward0 object at 0x...>
z.grad_fn: <AddBackward0 object at 0x...>
```

## The Computation Graph

Each operation creates a node in the computation graph:

```
Forward:
x (leaf) → [PowBackward0] → y → [AddBackward0] → z

Backward (when z.backward() is called):
z ← [AddBackward0] ← y ← [PowBackward0] ← x
    grad = 1         grad = 1    grad = 2*x
```

## Key Concepts

### 1. **Leaf Tensors**
- Created directly by user (not from operations)
- `grad_fn` is `None`
- Only leaf tensors accumulate gradients in `.grad`

```python
x = torch.tensor([2.0], requires_grad=True)  # Leaf tensor
print(x.is_leaf)  # True
print(x.grad_fn)  # None
```

### 2. **Non-Leaf Tensors**
- Result of operations
- Have `grad_fn` pointing to their creator
- Don't accumulate gradients by default

```python
y = x * 2  # Non-leaf tensor
print(y.is_leaf)  # False
print(y.grad_fn)  # <MulBackward0>
```

### 3. **grad_fn Chains**
Each `grad_fn` has `next_functions` pointing to previous operations:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 2
z = y + 3

print(z.grad_fn)  # <AddBackward0>
print(z.grad_fn.next_functions)  
# ((<MulBackward0>, 0), (<AccumulateGrad>, 0))
#    ^-- previous operation  ^-- gradient accumulator for constant
```

## How `.backward()` Works

When you call `loss.backward()`:

1. **Start from loss tensor**
   ```python
   loss.grad_fn  # Starting point
   ```

2. **Walk the graph backward**
   - Calls `.backward()` on each `grad_fn`
   - Each `grad_fn` computes gradients for its inputs
   - Passes gradients to `next_functions`

3. **Accumulate at leaf tensors**
   - When reaching a leaf tensor, gradient is stored in `.grad`

## Code Pointers in PyTorch

### C++ Implementation

The core autograd engine is in C++:

**Main autograd engine:**
- `torch/csrc/autograd/engine.cpp` - Executes backward pass
- `torch/csrc/autograd/function.cpp` - Base class for all `grad_fn`
- `torch/csrc/autograd/functions/` - Specific backward functions

**Key functions:**
```cpp
// torch/csrc/autograd/engine.cpp
auto Engine::execute(...) {
    // Main backward execution loop
    // Walks the graph and calls grad_fn->apply()
}
```

### Python Side

**Tensor class:**
- `torch/_tensor.py` - Defines `Tensor.backward()`
- `torch/autograd/__init__.py` - Main autograd interface

**Custom autograd functions:**
- `torch/autograd/function.py` - `torch.autograd.Function` base class
- This is what `CompiledFunction` inherits from!

## Creating Custom grad_fn

You can create custom backward functions:

```python
class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * 2 * x

# Usage
x = torch.tensor([2.0], requires_grad=True)
y = MySquare.apply(x)
print(y.grad_fn)  # <MySquareBackward object>

y.backward()
print(x.grad)  # tensor([4.0]) = 2 * 2
```

## Visualizing the Graph

### Using `torchviz`:
```python
from torchviz import make_dot

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y + 3

make_dot(z, params={'x': x}).render('computation_graph', format='png')
```

### Manual inspection:
```python
def print_graph(tensor, indent=0):
    print(' ' * indent + str(tensor.grad_fn))
    if tensor.grad_fn is not None:
        for next_fn, _ in tensor.grad_fn.next_functions:
            if next_fn is not None:
                print_graph(next_fn, indent + 2)

print_graph(z)
```

Output:
```
<AddBackward0 object at 0x...>
  <PowBackward0 object at 0x...>
    <AccumulateGrad object at 0x...>
  <AccumulateGrad object at 0x...>
```

## Debugging Autograd

### 1. **Check if gradients are computed**
```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 2
z = y + 3

print(x.requires_grad)  # True
print(y.requires_grad)  # True
print(z.requires_grad)  # True
```

### 2. **Retain gradients for non-leaf tensors**
```python
y.retain_grad()  # Keep gradient for non-leaf
z.backward()
print(y.grad)  # Now available!
```

### 3. **Detect in-place operations**
```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 2
y += 1  # Error! In-place op on tensor used in backward
```

### 4. **Hooks for debugging**
```python
def hook_fn(grad):
    print(f"Gradient: {grad}")
    
x = torch.tensor([2.0], requires_grad=True)
x.register_hook(hook_fn)

y = x ** 2
y.backward()
# Prints: Gradient: tensor([4.0])
```

## Connection to CompiledFunction

In our earlier example, `CompiledFunction` is just another `grad_fn`:

```python
class CompiledFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        # Run compiled forward
        return compiled_fw(args)
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        # Run compiled backward
        return compiled_bw(ctx.saved_tensors, grad_outputs)
```

When you use `torch.compile()`:
```python
output = compiled_model(input)
print(output.grad_fn)  # <CompiledFunctionBackward>
```

The output tensor's `grad_fn` points to `CompiledFunction.backward()`, which runs the compiled backward graph!

## Learning Resources

### Official PyTorch Docs
1. **Autograd mechanics**: https://pytorch.org/docs/stable/notes/autograd.html
2. **Extending autograd**: https://pytorch.org/docs/stable/notes/extending.html
3. **Autograd API**: https://pytorch.org/docs/stable/autograd.html

### Code to Read
1. Start with examples:
   - `/home/lsakka/pytorch10/pytorch/test/test_autograd.py`
   - `/home/lsakka/pytorch10/pytorch/test/test_custom_ops.py`

2. Understand custom functions:
   - `/home/lsakka/pytorch10/pytorch/torch/autograd/function.py`

3. See how ops register backward:
   - `/home/lsakka/pytorch10/pytorch/torch/_ops.py`
   - `/home/lsakka/pytorch10/pytorch/tools/autograd/derivatives.yaml`

### Key Files in PyTorch Codebase
```
torch/csrc/autograd/
├── engine.cpp              # Main backward execution engine
├── function.cpp            # Base grad_fn class
├── functions/              # Backward implementations
│   ├── basic_ops.cpp      # Add, Mul, etc.
│   └── tensor.cpp         # Tensor operations
└── variable.cpp           # Tensor + autograd integration

torch/autograd/
├── function.py            # Python Function base class
├── grad_mode.py           # no_grad, enable_grad contexts
└── profiler.py            # Profiling autograd
```

## Quick Reference

| Concept | Meaning |
|---------|---------|
| `tensor.grad_fn` | Backward function that created this tensor |
| `tensor.is_leaf` | True if tensor is a leaf (user-created) |
| `tensor.requires_grad` | True if tensor tracks gradients |
| `tensor.grad` | Accumulated gradient (only for leaf tensors) |
| `tensor.retain_grad()` | Keep gradient for non-leaf tensor |
| `tensor.backward()` | Compute gradients for this tensor |
| `next_functions` | Previous operations in the graph |
| `AccumulateGrad` | Final node that accumulates gradient to `.grad` |

## Summary

**grad_fn** is the backbone of PyTorch's autograd system. It:
- Links tensors to their creator operations
- Forms the computation graph
- Enables automatic differentiation
- Gets called automatically during `.backward()`
- Is how compiled backward works - `CompiledFunction.backward` is just another `grad_fn`!

Understanding `grad_fn` is key to understanding how PyTorch computes gradients automatically!
