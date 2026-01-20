# Wrapper and CUDA Graph Pythonify Contributor Guide

This document describes how pythonify models and emits AOTAutograd post-compile
wrappers and CUDA graph capture/replay. It is intended for contributors who need
to understand, extend, or debug wrapper handling in the pythonify pipeline.

For general pythonify architecture and how to add new IR nodes, see `README.md`.
For the detailed AOTAutograd wrapper audit, see `wrapper_audit.md`.

## Table of Contents

1. [Overview](#overview)
2. [Wrapper Modeling System](#wrapper-modeling-system)
3. [CUDA Graph Pythonify Behavior](#cuda-graph-pythonify-behavior)
4. [Updating Tests When Wrapper Stack Changes](#updating-tests-when-wrapper-stack-changes)
5. [Common Debugging Scenarios](#common-debugging-scenarios)

---

## Overview

When `torch.compile` executes, AOTAutograd applies a stack of post-compile
wrappers that modify calling conventions, handle input/output transformations,
and manage autograd assembly. Pythonify must model these wrappers to generate
Python code that faithfully reproduces the compiled function's behavior.

The wrapper handling flows through three stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. METADATA CAPTURE (compile_fx.py / convert_frame.py)                      │
│    - Extract wrapper metadata from TracingContext.fw_metadata               │
│    - Populate CompilationArtifacts.wrapper_stack_order/metadata             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. IR CONSTRUCTION (pipeline.py)                                            │
│    - _build_wrapper_nodes() creates IR nodes from wrapper metadata          │
│    - Wrappers are processed in segment order: forward_inference →           │
│      autograd_assembly → dispatch                                           │
│    - Each wrapper type maps to a specific IRNode subclass                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. CODE GENERATION (gen_python.py / gen_binary.py)                          │
│    - Python backend: emit helper functions that mirror wrapper behavior     │
│    - Binary backend: ignore wrapper nodes (runtime wrappers already exist)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Wrapper Modeling System

### Wrapper IR Node Types

Each AOTAutograd wrapper has a corresponding IR node class defined in `ir.py`:

| AOTAutograd Wrapper             | IR Node Class                      | Key Metadata                                |
|---------------------------------|------------------------------------|---------------------------------------------|
| EffectTokensWrapper             | `EffectTokensWrapperNode`          | `token_count`                               |
| AOTDispatchSubclassWrapper      | `AOTDispatchSubclassWrapperNode`   | `subclass_inp_meta`, `subclass_fw_graph_out_meta`, `num_fw_outs_saved_for_bw`, `maybe_subclass_meta` |
| FunctionalizedRngRuntimeWrapper | `FunctionalizedRngRuntimeWrapperNode` | `is_rng_op_functionalized`, `num_outputs_rng_offset`, `num_forward_returns`, `num_graphsafe_rng_states` |
| FakifiedOutWrapper              | `FakifiedOutWrapperNode`           | `out_metas`, `fwd_output_strides`           |
| RuntimeWrapper                  | `RuntimeWrapperNode`               | `indices_of_inps_to_detach`, `disable_amp`, `runtime_metadata` |
| AOTDedupeWrapper                | `AOTDedupeWrapperNode`             | `keep_arg_mask`, `add_dupe_map`, `needs_post_compile` |
| AOTSyntheticBaseWrapper         | `AOTSyntheticBaseWrapperNode`      | `synthetic_base_info`, `aliased_arg_idx_with_metadata_mutations`, `needs_post_compile` |
| DebugAssertWrapper              | `DebugAssertWrapperNode`           | `flat_requires_grad`                        |

### Wrapper Ordering

The wrappers are applied in a specific order defined by AOTAutograd. When pythonify
builds the IR, it must respect this order. The `wrapper_stack_order` field in
`CompilationArtifacts` defines the segments:

```
SEGMENT ORDER (inner-most callable first):
1. forward_inference - Wrappers around compiled forward/inference callable
   - EffectTokensWrapper
   - AOTDispatchSubclassWrapper
   - FunctionalizedRngRuntimeWrapper
   - FakifiedOutWrapper

2. autograd_assembly - Wrappers for autograd fw/bw stitching
   - RuntimeWrapper
   - DebugAssertWrapper (when config.debug_assert is enabled)

3. dispatch - Dispatch-level wrappers (applied in reverse order)
   - AOTSyntheticBaseWrapper
   - AOTDedupeWrapper
```

When generating Python code, the wrappers must be applied in reverse order
(outer-most first) to match how AOTAutograd's `post_compile` unwraps them.

### Metadata Flow

Wrapper metadata flows through the system as follows:

1. **During compilation** (`compile_fx.py`):
   - `_extract_wrapper_metadata_for_pythonify()` extracts metadata from
     `TracingContext.fw_metadata`
   - Metadata is attached to the inductor output dictionary

2. **Building artifacts** (`convert_frame.py`):
   - `_build_pythonify_artifacts()` extracts `wrapper_stack_order` and
     `wrapper_stack_metadata` from inductor output
   - Fields are passed to `CompilationArtifacts` constructor

3. **Context merging** (`context.py`):
   - `PythonifyContext.add_compilation_artifacts()` preserves wrapper metadata
   - `_merge_inductor_outputs()` merges metadata without loss

4. **Pipeline construction** (`pipeline.py`):
   - `RuntimeWrapperPipeline._build_wrapper_nodes()` creates IR nodes
   - `_create_wrapper_node()` maps wrapper types to node classes

### Adding a New Wrapper

If a new AOTAutograd wrapper is added, follow these steps:

1. **Add IR node class** in `ir.py`:
   ```python
   @dataclass
   class NewWrapperNode(IRNode):
       """Docstring describing the wrapper's purpose."""
       # Required metadata fields
       metadata_field: Any = None

       def accept(self, visitor: "CodeGenVisitor") -> Any:
           return visitor.visit_new_wrapper(self)
   ```

2. **Add visitor methods** to `CodeGenVisitor` in `ir.py`:
   ```python
   @abc.abstractmethod
   def visit_new_wrapper(self, node: NewWrapperNode) -> Any:
       """Process a new wrapper node."""
       pass
   ```

3. **Implement in gen_python.py**:
   - Add `visit_new_wrapper()` method that emits helper functions
   - Emit code that mirrors the wrapper's runtime behavior

4. **Implement in gen_binary.py**:
   - Add `visit_new_wrapper()` method that returns `None`
   - Binary backend ignores wrapper nodes (runtime already handles them)

5. **Update pipeline.py**:
   - Add wrapper type to `_create_wrapper_node()` mapping
   - Ensure correct segment assignment in wrapper_stack_order

6. **Update compile_fx.py**:
   - Extract new metadata in `_extract_wrapper_metadata_for_pythonify()`

7. **Add tests** (see [Updating Tests](#updating-tests-when-wrapper-stack-changes))

---

## CUDA Graph Pythonify Behavior

### Overview

CUDA graphs capture a sequence of GPU operations and replay them with minimal
CPU overhead. Pythonify generates Python code that performs:

1. **Graph capture** - Record GPU operations during a warmup run
2. **Static buffer management** - Handle inputs that don't change between calls
3. **Graph replay** - Execute the captured graph on new inputs

### CUDA Graph IR Node

The `CUDAGraphSetupNode` in `ir.py` models CUDA graph configuration:

```python
@dataclass
class CUDAGraphSetupNode(IRNode):
    graph_id: str                           # Unique graph identifier
    warmup_runs: int = 1                    # Warmups before capture
    capture_mode: str = "thread_local"      # CUDA capture mode
    stream_name: str = "default"            # CUDA stream name
    pool_id: Optional[str] = None           # Memory pool for allocation
    static_inputs: bool = False             # All inputs are static
    static_input_indices: list[int]         # Indices of static inputs
    phase: CUDAGraphPhase                   # INFERENCE, FORWARD, or BACKWARD
    backward_graph_id: Optional[str]        # ID of paired backward graph
    saved_tensor_indices: list[int]         # Tensors saved for backward
    num_forward_outputs: Optional[int]      # Outputs (excluding saved)
    device_index: Optional[int]             # GPU device index
    skip_dynamic_graphs: bool = False       # Skip for dynamic shapes
```

### Phases

CUDA graphs support three phases:

| Phase       | Description                                      |
|-------------|--------------------------------------------------|
| `INFERENCE` | Single graph for inference (no backward)         |
| `FORWARD`   | Forward pass graph for training                  |
| `BACKWARD`  | Backward pass graph using saved tensors          |

For training, forward and backward are captured as separate graphs. The forward
graph's saved tensors are stored in static buffers accessible to the backward graph.

### Generated Code Structure

The Python codegen (`gen_python.py`) emits:

1. **Module-level state** - CUDA graph objects and static buffers:
   ```python
   _cuda_graph_inference = None
   _static_inputs_inference = None
   _captured_shapes_inference = None
   ```

2. **Warmup and capture logic** - First-call graph capture:
   ```python
   if _cuda_graph_inference is None:
       # Warmup runs
       for _ in range(warmup_runs):
           output = compiled_fn(*inputs)
       # Capture
       _cuda_graph_inference = torch.cuda.CUDAGraph()
       with torch.cuda.graph(_cuda_graph_inference):
           _static_outputs = compiled_fn(*_static_inputs)
   ```

3. **Replay wrapper** - Copy inputs and replay:
   ```python
   def _replay_cuda_graph(*args):
       # Copy dynamic inputs to static buffers
       for i, arg in enumerate(args):
           if i not in static_input_indices:
               _static_inputs[i].copy_(arg)
       # Replay graph
       _cuda_graph_inference.replay()
       return _static_outputs.clone()
   ```

### Dynamic Shape Handling

When `skip_dynamic_graphs=True`, the generated code checks input shapes:

```python
def _replay_cuda_graph(*args):
    current_shapes = [a.shape for a in args]
    if current_shapes != _captured_shapes:
        # Fall back to direct execution
        return compiled_fn(*args)
    # Normal replay path
    ...
```

### Training Mode

For training with CUDA graphs:

1. Forward graph captures the forward pass and saves tensors
2. Backward graph operates on the static saved tensors
3. The forward graph's `saved_tensor_indices` specify which outputs to save
4. The backward graph's `saved_tensor_indices` specify which inputs are saved tensors

---

## Updating Tests When Wrapper Stack Changes

### Test File Organization

Pythonify wrapper tests are organized by purpose:

| Test File                          | Purpose                                    |
|------------------------------------|--------------------------------------------|
| `test_pythonify_ir_wrappers.py`    | Unit tests for IR node creation/visitor    |
| `test_pythonify_wrapper_codegen.py`| Tests for generated Python helper code     |
| `test_pythonify_wrappers.py`       | End-to-end parity tests vs torch.compile   |
| `test_pythonify_cuda_graphs.py`    | CUDA graph codegen and parity tests        |
| `test_pythonify_binary.py`         | Binary backend handling of wrapper nodes   |
| `test_pythonify_context_pipeline.py`| Metadata flow through context/pipeline    |

### When to Update Tests

Update tests when:

1. **Adding a new wrapper** - Add tests in all relevant test files
2. **Changing wrapper metadata** - Update IR tests and codegen tests
3. **Changing wrapper order** - Update pipeline tests and parity tests
4. **Changing codegen output** - Update codegen verification tests

### Test Patterns

#### IR Node Unit Tests (`test_pythonify_ir_wrappers.py`)

Test node creation and visitor dispatch:

```python
def test_new_wrapper_node_creation(self):
    """Test NewWrapperNode creation with required metadata."""
    node = NewWrapperNode(
        metadata_field="value",
    )
    self.assertEqual(node.metadata_field, "value")

def test_new_wrapper_node_visitor_dispatch(self):
    """Test visitor dispatch for NewWrapperNode."""
    node = NewWrapperNode(metadata_field="value")
    visitor = MockVisitor()
    node.accept(visitor)
    self.assertTrue(visitor.visited_new_wrapper)
```

#### Codegen Unit Tests (`test_pythonify_wrapper_codegen.py`)

Test that generated helper functions work correctly:

```python
def test_new_wrapper_helper_function(self):
    """Test generated helper function for new wrapper."""
    # Create node with test metadata
    node = NewWrapperNode(metadata_field="test")

    # Generate code
    visitor = PythonCodeGenVisitor()
    visitor.visit_new_wrapper(node)
    code = visitor.get_generated_code()

    # Verify code is syntactically valid
    compile(code, "<test>", "exec")

    # Execute and verify behavior
    exec_globals = {"torch": torch}
    exec(code, exec_globals)
    helper_fn = exec_globals["_new_wrapper_helper"]
    result = helper_fn(test_input)
    self.assertEqual(result, expected_output)
```

#### Parity Tests (`test_pythonify_wrappers.py`)

Test that pythonified code matches torch.compile behavior:

```python
def test_new_wrapper_parity(self):
    """Test pythonified callable matches torch.compile for new wrapper."""
    model = ModelThatTriggersNewWrapper()

    # Compile normally
    compiled = torch.compile(model)
    compiled_output = compiled(test_input)

    # Compile with pythonify
    pythonified = torch.compile(model, pythonify=True)
    pythonified_output = pythonified(test_input)

    # Verify parity
    self.assertEqual(compiled_output, pythonified_output)
```

#### Binary Backend Tests (`test_pythonify_binary.py`)

Test that binary backend ignores wrapper nodes:

```python
def test_binary_ignores_new_wrapper(self):
    """Test binary codegen ignores NewWrapperNode."""
    node = NewWrapperNode(metadata_field="test")
    visitor = BinaryCodeGenVisitor()
    result = node.accept(visitor)
    self.assertIsNone(result)
```

### Running Tests

Run specific test files:

```bash
python test/dynamo/test_pythonify_ir_wrappers.py -v
python test/dynamo/test_pythonify_wrapper_codegen.py -v
python test/dynamo/test_pythonify_wrappers.py -v
python test/dynamo/test_pythonify_cuda_graphs.py -v
```

Run all pythonify tests:

```bash
python -m pytest test/dynamo/test_pythonify*.py -v
```

---

## Common Debugging Scenarios

### Wrapper Metadata Not Captured

**Symptom**: Generated code missing wrapper behavior

**Debug steps**:

1. Check `_extract_wrapper_metadata_for_pythonify()` in `compile_fx.py`
2. Verify `TracingContext.fw_metadata` has the expected fields
3. Add print statements to trace metadata flow:
   ```python
   # In compile_fx.py
   print(f"Extracted wrapper metadata: {wrapper_metadata}")

   # In convert_frame.py
   print(f"Building artifacts with: {wrapper_stack_order}")
   ```

### Wrong Wrapper Order

**Symptom**: Calling convention mismatch, incorrect inputs/outputs

**Debug steps**:

1. Review `wrapper_audit.md` for correct ordering
2. Check `_build_wrapper_nodes()` in `pipeline.py`
3. Verify segment order matches: forward_inference → autograd_assembly → dispatch

### Codegen Syntax Error

**Symptom**: `exec()` fails on generated code

**Debug steps**:

1. Save generated code to file: `pythonify="/tmp/debug.py"`
2. Manually inspect the file for syntax issues
3. Run `python -m py_compile /tmp/debug.py` to get line numbers
4. Check the relevant `visit_*` method in `gen_python.py`

### CUDA Graph Shape Mismatch

**Symptom**: CUDA graph replay fails with shape error

**Debug steps**:

1. Verify `skip_dynamic_graphs=True` is set for dynamic models
2. Check `static_input_indices` correctly identifies parameters/buffers
3. Ensure input tensors match the shapes used during capture

### Parity Test Failure

**Symptom**: Pythonified output differs from torch.compile

**Debug steps**:

1. Generate and save the Python code
2. Add print statements to compare intermediate values
3. Check if the wrapper is fully implemented (some have placeholders)
4. Verify wrapper metadata is being captured correctly

---

## Summary

The pythonify wrapper system faithfully models AOTAutograd's post-compile
wrappers through:

1. **IR nodes** (`ir.py`) - One node class per wrapper type
2. **Metadata capture** (`compile_fx.py`, `convert_frame.py`) - Extract from TracingContext
3. **Pipeline construction** (`pipeline.py`) - Build nodes in correct order
4. **Code generation** (`gen_python.py`) - Emit helper functions

When modifying wrappers:

1. Update the corresponding IR node class
2. Update metadata extraction
3. Update codegen visitor methods
4. Add/update tests in all relevant test files

Keep this documentation synchronized with changes to the wrapper system.
