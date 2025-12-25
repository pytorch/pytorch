# Precompilation API Refactoring Plan

## Overview

Transform the current monolithic precompilation flow (`ugly.py`) into a composable, phase-by-phase API (`beautiful.py`).

**Current Flow (ugly.py):**
```python
torch._dynamo.config.enable_aot_compile = True
compiled_model = torch.compile(model, fullgraph=True)
compiled_model.forward.aot_compile(((input_ids,), {})).save_compiled_function(path)
loaded_fn = torch.compiler.load_compiled_function(f)
```

**Target Flow (beautiful.py):**
```python
gm, bytecode, guards = torch.Precompile.dynamo(model, example_input)
joint_gm, guards = torch.Precompile.aot_autograd(gm, guards)
compiled_joint_gm, guards = torch.Precompile.inductor(joint_gm, guards)
precompiled_artifact = torch.Precompile.precompile(compiled_joint_gm)
torch.Precompile.save("/tmp/model.pt", precompiled_artifact, bytecode, guards)

precompiled_artifact = torch.Precompile.load("/tmp/model.pt")
precompiled_artifact(example_input)
```

---

## Design Decisions

1. **API Location:** `torch/precompile/` - A new standalone module
2. **Guards Handling:** Accumulate guards - Each phase adds to guards from previous phases
3. **Test Location:** Same directory as `ugly.py` and `beautiful.py` (pytorch root)

---

## Critical Files

| File | Purpose |
|------|---------|
| `torch/precompile/__init__.py` | **NEW** - Main Precompile module |
| `torch/precompile/types.py` | **NEW** - Dataclasses (DynamoOutput, AOTAutogradOutput, etc.) |
| `torch/_dynamo/aot_compile.py` | Current AOT compilation entry point |
| `torch/_dynamo/convert_frame.py` | Dynamo tracing & fullgraph_capture |
| `torch/_dynamo/aot_compile_types.py` | SerializableCallable interface |
| `torch/_dynamo/guards.py` | Guard creation & management |
| `torch/_functorch/_aot_autograd/` | AOT Autograd implementation |
| `torch/_inductor/compile_fx.py` | Inductor backend entry point |
| `torch/compiler/__init__.py` | Public API (load_compiled_function) |

---

## TODOs

### Phase 1: Create torch.Precompile Module Structure

#### TODO 1.1: Create the Precompile module skeleton ✅ DONE
Create `torch/precompile/__init__.py` with the `Precompile` class skeleton exposing the target API.

**Verification file:** `test_precompile_skeleton.py` ✅ (8 tests passing)
- Updated `test_unimplemented_methods_raise_not_implemented` → `test_all_methods_are_implemented` to reflect that all methods are now fully implemented (not stubs)
```python
# Should verify:
# - torch.Precompile exists
# - torch.Precompile.dynamo, aot_autograd, inductor, precompile, save, load are callable
```

---

### Phase 2: Implement torch.Precompile.dynamo()

#### TODO 2.1: Define DynamoOutput dataclass ✅ DONE
Create a dataclass to hold dynamo output:
- `graph_module: torch.fx.GraphModule`
- `bytecode: types.CodeType`
- `guards: GuardsState` (serializable guards)
- `example_inputs: list[torch.Tensor]`
- `fake_mode: FakeTensorMode`

**Verification file:** `test_dynamo_output_dataclass.py` ✅ (6 tests passing)

#### TODO 2.2: Extract dynamo tracing from fullgraph_capture ✅ DONE
Refactor `convert_frame.fullgraph_capture()` to be callable standalone via `Precompile.dynamo()`.

Key changes:
- Extract the `compile_frame()` call logic
- Return `DynamoOutput` instead of `CaptureOutput`
- Thread guards explicitly

**Verification file:** `test_precompile_dynamo.py` ✅ (8 tests passing)
```python
# Should verify:
# - Precompile.dynamo(model, example_input) returns DynamoOutput
# - DynamoOutput.graph_module is a valid FX graph
# - DynamoOutput.bytecode is a code object
# - DynamoOutput.guards is serializable
```

---

### Phase 3: Implement torch.Precompile.aot_autograd()

#### TODO 3.1: Define AOTAutogradOutput dataclass ✅ DONE
Create a dataclass to hold aot_autograd output:
- `joint_graph: torch.fx.GraphModule` (forward + backward)
- `guards: GuardsState`
- `metadata: AOTConfig` (for inductor)

**Verification file:** `test_aot_autograd_output_dataclass.py` ✅ (6 tests passing)

#### TODO 3.2: Expose aot_autograd as standalone API ✅ DONE
Extract the AOT autograd transformation from `aot_module_simplified` to be callable via `Precompile.aot_autograd()`.

Key changes:
- Take `DynamoOutput` as input
- Return `AOTAutogradOutput`
- Pass through guards

**Verification file:** `test_precompile_aot_autograd.py` ✅ (8 tests passing)

---

### Phase 4: Implement torch.Precompile.inductor()

#### TODO 4.1: Define InductorOutput dataclass ✅ DONE
Create a dataclass to hold inductor output:
- `compiled_module: CompiledFxGraph` or similar
- `guards: GuardsState`
- `kernel_artifacts: bytes` (serialized kernels)

**Verification file:** `test_inductor_output_dataclass.py` ✅ (6 tests passing)

#### TODO 4.2: Expose inductor compilation as standalone API ✅ DONE
Extract inductor compilation from `compile_fx` to be callable via `Precompile.inductor()`.

Key changes:
- Take `AOTAutogradOutput` as input
- Return `InductorOutput`
- Pass through guards
- Extract placeholder inputs from joint graph metadata
- Note: Currently supports inference mode (trace_joint=False); full joint training graph support requires partitioning

**Verification file:** `test_precompile_inductor.py` ✅ (8 tests passing)

---

### Phase 5: Implement torch.Precompile.precompile()

#### TODO 5.1: Define PrecompiledArtifact dataclass ✅ DONE
Create a dataclass to bundle all compilation artifacts:
- `inductor_output: InductorOutput`
- `runtime_env: GraphRuntimeEnv`
- `signature: inspect.Signature`
- `system_info: SystemInfo`

**Verification file:** `test_precompiled_artifact_dataclass.py` ✅ (14 tests passing)

#### TODO 5.2: Implement precompile() to bundle artifacts ✅ DONE
Combine all previous outputs into a single `PrecompiledArtifact`.

**Verification file:** `test_precompile_precompile.py` ✅ (8 tests passing)
```python
# Should verify:
# - Precompile.precompile(inductor_output) returns PrecompiledArtifact
# - PrecompiledArtifact is callable
```

---

### Phase 6: Implement torch.Precompile.save() and load()

#### TODO 6.1: Implement save() serialization ✅ DONE
Serialize `PrecompiledArtifact`, bytecode, and guards to a file.

Use the existing serialization infrastructure from `AOTCompiledFunction.serialize()`.

**Verification file:** `test_precompile_save.py` ✅ (8 tests passing)
```python
# Should verify:
# - Precompile.save(path, artifact, bytecode, guards) creates a file
# - File is non-empty and loadable
```

#### TODO 6.2: Implement load() deserialization ✅ DONE
Deserialize from file and return a callable `PrecompiledArtifact`.

**Verification file:** `test_precompile_load.py` ✅ (9 tests passing)
```python
# Should verify:
# - Precompile.load(path) returns a callable
# - Loaded artifact produces correct output
```

---

### Phase 7: End-to-End Integration

#### TODO 7.1: Create integration test with beautiful.py flow ✅ DONE
Verify the complete flow works end-to-end.

**Verification file:** `test_precompile_e2e.py` ✅ (8 tests passing)
```python
# Should verify the complete flow from beautiful.py:
model = MyModel()
example_input = torch.randn(42)

gm, bytecode, guards = torch.Precompile.dynamo(model, example_input)
joint_gm, guards = torch.Precompile.aot_autograd(gm, guards)
compiled_joint_gm, guards = torch.Precompile.inductor(joint_gm, guards)
precompiled_artifact = torch.Precompile.precompile(compiled_joint_gm)
torch.Precompile.save("/tmp/model.pt", precompiled_artifact, bytecode, guards)

loaded = torch.Precompile.load("/tmp/model.pt")
result = loaded(example_input)
# Compare with eager model output
```

#### TODO 7.2: Verify backward compatibility ✅ DONE
Ensure existing `torch.compile` + `aot_compile` flow still works.

**Verification file:** `test_precompile_backward_compat.py` ✅ (8 tests passing)
```python
# Should verify ugly.py flow still works unchanged
```

---

### Phase 8: Expose via torch namespace

#### TODO 8.1: Add torch.Precompile to torch/__init__.py ✅ DONE
Expose the new API at `torch.Precompile`.

**Verification file:** `test_torch_precompile_import.py` ✅ (8 tests passing)
```python
# Should verify:
# - import torch; torch.Precompile exists
# - All methods accessible
```

---

## Summary of Verification Files

All test files are placed in the pytorch root directory (same as `ugly.py` and `beautiful.py`):

| TODO | Verification File |
|------|-------------------|
| 1.1 | `test_precompile_skeleton.py` |
| 2.1 | `test_dynamo_output_dataclass.py` |
| 2.2 | `test_precompile_dynamo.py` |
| 3.1 | `test_aot_autograd_output_dataclass.py` |
| 3.2 | `test_precompile_aot_autograd.py` |
| 4.1 | `test_inductor_output_dataclass.py` |
| 4.2 | `test_precompile_inductor.py` |
| 5.1 | `test_precompiled_artifact_dataclass.py` |
| 5.2 | `test_precompile_precompile.py` |
| 6.1 | `test_precompile_save.py` |
| 6.2 | `test_precompile_load.py` |
| 7.1 | `test_precompile_e2e.py` |
| 7.2 | `test_precompile_backward_compat.py` |
| 8.1 | `test_torch_precompile_import.py` |
