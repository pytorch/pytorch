---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

(higher_order_ops)=

# Control Flow Operators

Control flow operators are structured operators in PyTorch that enable expressing complex control flow patterns in a way that is compatible with `torch.compile` and `torch.export`. Unlike regular Python control flow, these operators preserve their semantics through torch.compile and torch.export, enabling data-dependent control flow in traced programs.

```{warning}
Control flow operators are prototype features in PyTorch. They may have limited support for certain
input/output types and some may not fully support training. Read more about feature classification at:
https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## Why Use Control Flow Operators?

PyTorch lets you write models using native Python, including control flow like `if` statements, `for` loops, and `while` loops. This is great for flexibility, but it creates challenges for compilation.

Consider this simple example:

```python
if mod.static_config == 0:
    return f(x)
return g(x)
```

The two branches might have completely different operations. If we tried to compile both branches every time we hit an `if` statement, the number of code paths would explode exponentially and quickly become intractable.

To deal with this, `torch.compile` uses **specialization and guards**. When tracing your model, the compiler picks one code path based on the current value of the predicate (specialization), then adds a guard to check that assumption at runtime. If the guard fails, it recompiles.

The same applies to loops: the compiler unrolls them and guards on the number of iterations. This produces a straight-line computational graph that's easy to optimize.

This approach works well for static control flow, but breaks down in several cases:

- **Data-dependent control flow**: When the predicate depends on the *value* of a tensor, the compiler can't pick a branch at compile time because the value isn't known yet. Similarly, it can't unroll a `while` loop if the iteration count depends on tensor values. The compiler handles this by graph breaking and falling back to Python, which also makes it impossible to run the model without a Python runtime (e.g., on edge devices).

- **Dynamic shape-dependent control flow**: When the number of loop iterations or the branch predicate depends on a dynamic tensor size, specialization means the compiled code only works for that specific size. The compiler has to recompile whenever the size changes.

- **Large computational graphs**: Even with a static iteration count, unrolling a large loop creates a graph that grows linearly with the number of iterations, even though each iteration does the same thing. This leads to long compile times and high memory usage.

Control flow operators solve these problems by representing control flow as explicit operators that the compiler understands. Instead of specializing away the control flow, these operators preserve it in the compiled graph.

## Available Operators

```{toctree}
:maxdepth: 1

cond
while_loop
scan
associative_scan
map
```

### Quick Comparison

| Operator | Use Case | Example |
|----------|----------|---------|
| [cond](cond.md) | If `pred` is True, returns `true_fn(*operands)`, otherwise returns `false_fn(*operands)`. | `cond(pred, true_fn, false_fn, operands)` |
| [while_loop](while_loop.md) | While `cond_fn(*operands)` is True, executes `body_fn(*operands)`, which returns the operands for the next iteration. | `while_loop(cond_fn, body_fn, operands)` |
| [scan](scan.md) | Applies cumulative operations to `xs` with carried state | `scan(combine_fn, init, xs)` |
| [associative_scan](associative_scan.md) | Similar to `scan`, but requiring an associative `combine_fn` to allow for more optimizations. | `associative_scan(add, xs, dim=0)` |
| [map](map.md) | Computes `fn` on each slice of `xs` and returns the stacked output. | `map(fn, xs)` |
