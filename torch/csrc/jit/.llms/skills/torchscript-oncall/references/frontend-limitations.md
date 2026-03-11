# Frontend Limitations

## Table of Contents

1. [Unsupported Python Constructs](#unsupported-python-constructs)
2. [Type System Limitations](#type-system-limitations)
3. [Module and Compilation Issues](#module-and-compilation-issues)
4. [Compatibility Gaps](#compatibility-gaps)

---

## Unsupported Python Constructs

These are inherent limitations of TorchScript's Python subset.

### try/catch blocks (6+ instances)

Most frequently encountered limitation. Blocks usage of `justknobs_check`, Python `inspect`/`traceback`, assertion utilities, and any stdlib function using exception handling.

**Workaround:** Write custom C++ ops, use `@torch.jit.ignore`, or restructure code to avoid try/catch.

### *args / **kwargs (4+ instances)

TorchScript cannot compile variadic arguments. Common triggers: `nn.Module.to(*args, **kwargs)`, `functional_call`, calibration models with variadic signatures.

**Workaround:** Rewrite with explicit parameter lists; use `List`/`Dict` instead.

### Lambda functions (2+ instances)

`nn.Module.cuda()` internally uses a lambda (`self._apply(lambda t: t.cuda(device))`), causing compilation failure.

**Workaround:** Use `tensor.to(device=device)` directly instead of `module.cuda()`.

### Other unsupported constructs

| Construct | Notes |
|---|---|
| Inner function definitions | Raises `UnsupportedNodeError` |
| `Callable` in containers | `List[Callable]`, `Dict[str, Callable]` unsupported |
| `typing.Mapping` | Only `Dict` is supported |
| `datetime`, `inspect`, `traceback`, `logger` | Standard library modules unsupported |
| `print()` | Works, but `logger.info()` does not |
| `warnings.warn()` | Works as alternative to `print()` |

---

## Type System Limitations

### Type refinement fails on `self.` member variables (3+ instances)

After `if isinstance(self.x, Tensor)` or `if self.x is not None`, TorchScript still treats `self.x` as `Optional[Tensor]`.

**Workaround:** Assign to a local variable first:
```python
x = self.x
if x is not None:
    # use x here
```

### Empty container type inference (3+ instances)

`self.my_list: List[str] = []` in `__init__` fails because TorchScript mostly ignores `__init__` for type inference.

**Workaround:** Annotate at the class level:
```python
class MyModule(nn.Module):
    my_list: List[str]
```

### Union type narrowing (2+ instances)

Iterating over `Union[List[Tensor], List[Union[Tensor, List[Tensor]]]]` fails. Use carefully structured nested `if` blocks with `torch.jit.isinstance`.

### nn.Module as a type (3+ instances)

- Cannot return `nn.Module` types from functions
- Module types are name-mangled per-instance, incompatible across compilation units
- Subclassing is officially unsupported (works in practice if you avoid `super()`)

**Workaround:** Use `@torch.jit.interface` for polymorphism (but even this doesn't support returning modules).

### torch.jit.unused/ignore not fully effective (2+ instances)

Even when marking methods as `unused` or `ignore`, TorchScript may still compile the containing class, failing on varargs or unsupported patterns.

**Workaround:** `torch.jit._drop` works as a stronger alternative in some cases.

---

## Module and Compilation Issues

### "Can't redefine method: forward" (3+ instances)

Caused by TorchScript's global compilation unit tracking class definitions. Scripting the same module class twice in a single process triggers this. Also triggered by calling `trace_encoder()` inside `__init__`.

### Dynamic ModuleDict/ModuleList indexing (2+ instances)

`torch.jit.freeze` does not support `prim::ModuleContainerIndex`. `ModuleDict` keys must all exist at script time; dynamic key access fails.

### Hook support is incomplete

`register_module_forward_hook` and `register_module_forward_pre_hook` do not work on `ScriptModule` (PyTorch issue #34329, never implemented).

### Short-circuit evaluation with is_scripting()

`if not torch.jit.is_scripting() and not is_fx_tracing()` fails because TorchScript does not short-circuit compound boolean expressions.

**Workaround:** Use nested `if` statements.

---

## Compatibility Gaps

### PT2/torch.compile interop (2+ instances)

- `torch._check` (PT2 export shape constraints) is not registered as a TorchScript op
- PT2.0 FX graph modules cannot be `jit.script`-ed (but `jit.trace` works)

### Python version sensitivity (2+ instances)

- Python 3.12: `Enum.__str__` handling broke
- Python 3.13: `inspect` behavior changes caused test failures

### ONNX export

Exporting ONNX from TorchScript models is poorly supported. Recommend exporting from `nn.Module` directly, bypassing TorchScript.
