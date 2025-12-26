# Precompilation API Migration Plan

This plan outlines the migration from the current "ugly" precompilation flow (using `torch.compile` + `aot_compile` + `save_compiled_function` + `load_compiled_function`) to the new "beautiful" flow (using `torch.Precompile.dynamo()` + `torch.Precompile.aot_autograd()` with regional inductor).

## Summary of Changes

**Current (ugly.py):**
- Uses `torch._dynamo.config.enable_aot_compile = True`
- Uses `torch.compile()` followed by `.aot_compile()` and `.save_compiled_function()`
- Loads with `torch.compiler.load_compiled_function()`

**Target (beautiful.py):**
- Uses `torch.Precompile.dynamo()` to trace and capture FX graph
- Uses `torch.Precompile.aot_autograd()` with custom compiler (e.g., `regional_inductor`)
- Supports `fx_traceback.annotate()` for marking regions for selective compilation

---

## TODOs

### Phase 1: Implement `torch.Precompile` Class

- [x] **TODO 1: Create the `torch.Precompile` class stub**
  - Create a new class `Precompile` that will serve as the namespace for the new precompilation API
  - Location: Add to `torch/_dynamo/precompile.py` or similar
  - **Verification test:** `tests/test_todo_01_precompile_class.py`
    ```python
    # Test that torch.Precompile exists and is a class
    import torch
    assert hasattr(torch, 'Precompile'), "torch.Precompile should exist"
    ```

---

- [x] **TODO 2: Implement `Precompile.dynamo()` signature**
  - Implement the static method signature: `dynamo(model, *example_inputs) -> Tuple[GraphModule, bytecode, guards, example_inputs]`
  - This should accept a model and example inputs and return the traced graph
  - **Verification test:** `tests/test_todo_02_dynamo_signature.py`
    ```python
    # Test that Precompile.dynamo exists and is callable
    import torch
    assert hasattr(torch.Precompile, 'dynamo'), "Precompile.dynamo should exist"
    assert callable(torch.Precompile.dynamo), "Precompile.dynamo should be callable"
    ```

---

- [x] **TODO 3: Implement `Precompile.dynamo()` tracing logic**
  - Use `torch._dynamo.export()` or similar internal APIs to trace the model
  - Capture the FX GraphModule, bytecode, and guards
  - Return example inputs in the format expected by downstream phases
  - **Verification test:** `tests/test_todo_03_dynamo_tracing.py`
    ```python
    # Test that Precompile.dynamo can trace a simple model
    import torch
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def forward(self, x):
            return x * 2

    model = SimpleModel()
    example_input = torch.randn(2, 3)
    gm, bytecode, guards, example_inputs = torch.Precompile.dynamo(model, example_input)

    assert gm is not None, "GraphModule should be returned"
    assert hasattr(gm, 'graph'), "GraphModule should have a graph attribute"
    ```

---

- [x] **TODO 4: Implement `Precompile.aot_autograd()` signature**
  - Implement the static method signature: `aot_autograd(gm, guards, compiler=None) -> Tuple[compiled_fn, guards]`
  - This should accept a GraphModule and guards, and return a compiled function
  - **Verification test:** `tests/test_todo_04_aot_autograd_signature.py`
    ```python
    # Test that Precompile.aot_autograd exists and is callable
    import torch
    assert hasattr(torch.Precompile, 'aot_autograd'), "Precompile.aot_autograd should exist"
    assert callable(torch.Precompile.aot_autograd), "Precompile.aot_autograd should be callable"
    ```

---

- [x] **TODO 5: Implement `Precompile.aot_autograd()` compilation logic**
  - Use `torch._functorch.aot_autograd` or similar to generate forward/backward graphs
  - Accept an optional `compiler` argument (default to inductor)
  - Return the compiled callable and updated guards
  - **Verification test:** `tests/test_todo_05_aot_autograd_compilation.py`
    ```python
    # Test end-to-end Precompile.dynamo -> Precompile.aot_autograd
    import torch
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def forward(self, x):
            return x * 2 + 1

    model = SimpleModel()
    example_input = torch.randn(2, 3)
    gm, bytecode, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
    compiled_fn, guards = torch.Precompile.aot_autograd(gm, guards)

    assert callable(compiled_fn), "compiled_fn should be callable"
    output = compiled_fn(example_inputs)
    expected = model(example_input)
    assert torch.allclose(output, expected), "Compiled output should match eager output"
    ```

---

### Phase 2: Implement Regional Inductor Support

- [x] **TODO 6: Verify `fx_traceback.annotate()` preserves metadata in traced graph**
  - Ensure that when using `fx_traceback.annotate({"compile_with_inductor": "region_name"})`, the metadata is preserved in the traced FX graph nodes
  - **Verification test:** `tests/test_todo_06_annotate_metadata.py`
    ```python
    # Test that fx_traceback.annotate metadata is preserved in traced graph
    import torch
    import torch.nn as nn
    from torch.fx import traceback as fx_traceback

    class AnnotatedModel(nn.Module):
        def forward(self, x):
            with fx_traceback.annotate({"compile_with_inductor": "test_region"}):
                y = x * 2
            return y + 1

    model = AnnotatedModel()
    example_input = torch.randn(2, 3)
    gm, _, _, _ = torch.Precompile.dynamo(model, example_input)

    # Check that at least one node has the annotation
    found_annotation = False
    for node in gm.graph.nodes:
        custom = node.meta.get('custom', None)
        if custom and 'compile_with_inductor' in custom:
            found_annotation = True
            break

    assert found_annotation, "Annotation should be preserved in traced graph"
    ```

---

- [x] **TODO 7: Implement/verify `regional_inductor` compiler exists**
  - Ensure `torch.fx.passes.regional_inductor.regional_inductor` exists and can be used as a compiler
  - This compiler should only compile nodes marked with `compile_with_inductor` annotation
  - **Verification test:** `tests/test_todo_07_regional_inductor_exists.py`
    ```python
    # Test that regional_inductor is importable and callable
    from torch.fx.passes.regional_inductor import regional_inductor
    assert callable(regional_inductor), "regional_inductor should be callable"
    ```

---

- [x] **TODO 8: Implement regional compilation in `Precompile.aot_autograd()` with custom compiler**
  - When a custom compiler (e.g., `regional_inductor`) is passed, use it instead of the default inductor
  - The compiler should receive the FX graph and return a compiled callable
  - **Verification test:** `tests/test_todo_08_custom_compiler.py`
    ```python
    # Test that Precompile.aot_autograd accepts custom compiler
    import torch
    import torch.nn as nn
    from torch.fx.passes.regional_inductor import regional_inductor

    class SimpleModel(nn.Module):
        def forward(self, x):
            return x * 2

    model = SimpleModel()
    example_input = torch.randn(2, 3)
    gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
    compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards, compiler=regional_inductor)

    assert callable(compiled_fn), "compiled_fn should be callable with custom compiler"
    ```

---

### Phase 3: Wire Up `torch.Precompile` to `torch` Namespace

- [x] **TODO 9: Export `Precompile` class from `torch` namespace**
  - Add `Precompile` to `torch/__init__.py` so it's accessible as `torch.Precompile`
  - **Verification test:** `tests/test_todo_09_torch_namespace.py`
    ```python
    # Test that torch.Precompile is accessible
    import torch
    assert hasattr(torch, 'Precompile'), "torch.Precompile should be accessible"
    assert hasattr(torch.Precompile, 'dynamo'), "torch.Precompile.dynamo should be accessible"
    assert hasattr(torch.Precompile, 'aot_autograd'), "torch.Precompile.aot_autograd should be accessible"
    ```

---

### Phase 4: End-to-End Integration Tests

- [x] **TODO 10: End-to-end test with simple model (forward only)**
  - Test the complete flow: `Precompile.dynamo()` -> `Precompile.aot_autograd()` -> run inference
  - Verify output matches eager execution
  - **Verification test:** `tests/test_todo_10_e2e_forward.py`
    ```python
    # End-to-end test: forward pass with Precompile API
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = MLP()
    example_input = torch.randn(4, 10)

    gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
    compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards)

    with torch.no_grad():
        compiled_out = compiled_fn(example_inputs)
        eager_out = model(example_input)

    assert torch.allclose(compiled_out, eager_out, atol=1e-5), "Forward pass should match"
    print("SUCCESS: Forward pass matches eager execution")
    ```

---

- [x] **TODO 11: End-to-end test with backward pass (training)**
  - Test that gradients flow correctly through the compiled function
  - Verify gradients match eager execution
  - **Verification test:** `tests/test_todo_11_e2e_backward.py`
    ```python
    # End-to-end test: backward pass with Precompile API
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    torch.manual_seed(42)
    model = MLP()
    example_input = torch.randn(4, 10)

    gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
    compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards)

    # Forward + backward with compiled
    compiled_out = compiled_fn(example_inputs)
    compiled_out.sum().backward()
    compiled_grads = {name: p.grad.clone() for name, p in model.named_parameters()}

    # Reset gradients
    model.zero_grad()

    # Forward + backward with eager
    eager_out = model(example_input)
    eager_out.sum().backward()
    eager_grads = {name: p.grad.clone() for name, p in model.named_parameters()}

    for name in compiled_grads:
        assert torch.allclose(compiled_grads[name], eager_grads[name], atol=1e-5), f"Gradient mismatch for {name}"

    print("SUCCESS: Backward pass gradients match eager execution")
    ```

---

- [x] **TODO 12: End-to-end test with regional inductor on Transformer**
  - Test the full beautiful.py flow with a Transformer model
  - Use `fx_traceback.annotate()` to mark attention and MLP regions
  - Verify output matches eager execution
  - **Verification test:** `tests/test_todo_12_e2e_transformer.py`
    ```python
    # End-to-end test: Transformer with regional inductor
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.fx import traceback as fx_traceback
    from torch.fx.passes.regional_inductor import regional_inductor

    class SimpleAttention(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            with fx_traceback.annotate({"compile_with_inductor": "attention"}):
                out = F.softmax(self.proj(x), dim=-1)
            return out

    class SimpleTransformer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = SimpleAttention(dim)
            self.mlp = nn.Linear(dim, dim)

        def forward(self, x):
            x = self.attn(x)
            with fx_traceback.annotate({"compile_with_inductor": "mlp"}):
                x = self.mlp(x)
            return x

    model = SimpleTransformer(32)
    example_input = torch.randn(2, 16, 32)

    gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
    compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards, compiler=regional_inductor)

    with torch.no_grad():
        compiled_out = compiled_fn(example_inputs)
        eager_out = model(example_input)

    assert torch.allclose(compiled_out, eager_out, atol=1e-4), "Transformer output should match"
    print("SUCCESS: Regional inductor Transformer matches eager execution")
    ```

---

### Phase 5: Documentation and Cleanup

- [x] **TODO 13: Add docstrings to `Precompile.dynamo()` and `Precompile.aot_autograd()`**
  - Document parameters, return values, and usage examples
  - **Verification test:** `tests/test_todo_13_docstrings.py`
    ```python
    # Test that docstrings are present
    import torch

    assert torch.Precompile.dynamo.__doc__ is not None, "Precompile.dynamo should have a docstring"
    assert torch.Precompile.aot_autograd.__doc__ is not None, "Precompile.aot_autograd should have a docstring"
    assert len(torch.Precompile.dynamo.__doc__) > 50, "Docstring should be substantive"
    assert len(torch.Precompile.aot_autograd.__doc__) > 50, "Docstring should be substantive"
    print("SUCCESS: Docstrings are present and substantive")
    ```

---

- [x] **TODO 14: Deprecate old `aot_compile` / `save_compiled_function` / `load_compiled_function` pattern**
  - Add deprecation warnings to the old API (if still needed for backwards compatibility)
  - Or document migration path in a separate doc
  - **Verification test:** `tests/test_todo_14_deprecation.py`
    ```python
    # Test that old API shows deprecation warning (if applicable)
    import warnings
    import torch
    import torch.nn as nn

    # This test is conditional - only run if old API is deprecated
    # For now, just verify the new API works as a replacement
    class SimpleModel(nn.Module):
        def forward(self, x):
            return x * 2

    model = SimpleModel()
    example_input = torch.randn(2, 3)

    # New API should work
    gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
    compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards)
    output = compiled_fn(example_inputs)

    print("SUCCESS: New Precompile API works as replacement for old pattern")
    ```

---

## Summary Checklist

| TODO | Description | Verification Test |
|------|-------------|-------------------|
| 1 | Create `torch.Precompile` class stub | `tests/test_todo_01_precompile_class.py` |
| 2 | Implement `Precompile.dynamo()` signature | `tests/test_todo_02_dynamo_signature.py` |
| 3 | Implement `Precompile.dynamo()` tracing logic | `tests/test_todo_03_dynamo_tracing.py` |
| 4 | Implement `Precompile.aot_autograd()` signature | `tests/test_todo_04_aot_autograd_signature.py` |
| 5 | Implement `Precompile.aot_autograd()` compilation logic | `tests/test_todo_05_aot_autograd_compilation.py` |
| 6 | Verify `fx_traceback.annotate()` preserves metadata | `tests/test_todo_06_annotate_metadata.py` |
| 7 | Implement/verify `regional_inductor` compiler | `tests/test_todo_07_regional_inductor_exists.py` |
| 8 | Implement regional compilation with custom compiler | `tests/test_todo_08_custom_compiler.py` |
| 9 | Export `Precompile` from `torch` namespace | `tests/test_todo_09_torch_namespace.py` |
| 10 | End-to-end test: forward pass | `tests/test_todo_10_e2e_forward.py` |
| 11 | End-to-end test: backward pass | `tests/test_todo_11_e2e_backward.py` |
| 12 | End-to-end test: Transformer with regional inductor | `tests/test_todo_12_e2e_transformer.py` |
| 13 | Add docstrings | `tests/test_todo_13_docstrings.py` |
| 14 | Deprecate old API pattern | `tests/test_todo_14_deprecation.py` |
