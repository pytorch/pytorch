# TorchFuzz - PyTorch Dynamic Shape Fuzzing Tool

TorchFuzz is an experimental fuzzing framework for testing PyTorch operations. It generates
random operation stacks, converts them to executable Python code, and tests them with both eager
mode and different configurations of `torch.compile()`

## Overview

TorchFuzz works by:
1. **Generating random tensor/scalar specifications** with various shapes, strides, and dtypes
2. **Creating operation stacks** that produce the target specification, by recursively fuzzing an op given a type spec
and fuzzing a valid type specs for the arguments. The operation is then repeated for each arg recursively, until a leaf
arg is reached (argument, constant, TODO(add reuse of existing variable)).
The result is a stack of pytorch operations to execute (TODO make it a graph instead of a stack).
3. **Converting operations stack to executable Python code**
4. **Testing both eager and compiled execution** The codegened code already have both eager and compiled versions

### Example Walkthrough

Here's a concrete example of how TorchFuzz generates a test:

**Target Spec Generated:** `TensorSpec(size=(2, 3), stride=(3, 1), dtype=torch.float32)`

**Operation Stack Created:**
```
Operation 0: torch.ops.aten.add -> TensorSpec(size=(2, 3), stride=(3, 1), dtype=torch.float32) (depth 0)
  └─ Operation 1: arg -> TensorSpec(size=(2, 3), stride=(3, 1), dtype=torch.float32) (depth 1)
  └─ Operation 2: torch.ops.aten.mul -> TensorSpec(size=(2, 3), stride=(3, 1), dtype=torch.float32) (depth 1)
      └─ Operation 3: arg -> TensorSpec(size=(2, 3), stride=(3, 1), dtype=torch.int32) (depth 2)
      └─ Operation 4: constant -> ScalarSpec(dtype=torch.float32) (depth 2)
```

**Generated Python Code:**
```python
import torch

def test_function(arg_0, arg_1):
    constant_0 = 2.5
    var_0 = torch.ops.aten.mul(arg_1, constant_0)
    var_1 = torch.ops.aten.add(arg_0, var_0)
    return var_1

# Test with both eager and compiled execution
result_eager = test_function(arg_0, arg_1)
result_compiled = torch.compile(test_function)(arg_0, arg_1)
assert torch.allclose(result_eager, result_compiled)
```


## Quick Start

### Single Test Run

```bash
cd tools/experimental/dynamic_shapes/torchfuzz
python fuzzer.py --single --seed 42
```
Note: Given a seed, the fuzzer is guanteed to generate the same program. (on the same gh commit).

### Continuous Fuzzing

```bash
python fuzzer.py --test --seed 1000 --max-depth 5
```

### With Debug Output

```bash
python fuzzer.py --single --log-level DEBUG --seed 42
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--single` | Run a single fuzz test (default: run continuous loop) |
| `--test` | Run continuous fuzzing loop |
| `--seed INT` | Set random seed for reproducible tests |
| `--max-depth INT` | Maximum operation stack depth (1-20) |
| `--log-level LEVEL` | Set logging level (DEBUG, INFO, WARNING, ERROR) |


### Core Components

1. **`tensor_fuzzer.py`** - Generates random tensor specifications (shapes, strides, dtypes)
2. **`ops_fuzzer.py`** - Creates operation stacks with type-aware operations
3. **`codegen.py`** - Converts operation stacks to executable Python code
4. **`fuzzer.py`** - Main orchestrator and CLI interface
5. **`visualize_stack.py`** - Creates visual diagrams of operation stacks

### Operation Types

**Tensor Operations:**
Write now the supported ops are very limited, its should be easy to extend that.
The fuzzer already very simple was able to cartch three bugs already.

- `torch.ops.aten.add` - Element-wise tensor addition
- `torch.ops.aten.mul` - Element-wise tensor multiplication
- `arg` - Function arguments (input tensors)

**Scalar Operations:**
- `scalar_add` - Python scalar addition
- `scalar_multiply` - Python scalar multiplication
- `torch.ops.aten.item` - Extract scalar from 1-element tensor
- `constant` - Generate constant values

## Example Output

```
Using seed: 42
Using max_depth: 3
⏱️  Step 1: Generating target spec...
   Completed in 0.001s - TensorSpec(size=(2, 3), stride=(3, 1), dtype=torch.float32)
⏱️  Step 2: Generating operation stack...
   Completed in 0.002s - 5 operations
⏱️  Step 3: Converting to Python code...
   Completed in 0.003s - 1247 chars
⏱️  Step 4: Executing Python code...
📄 Generated code written to: /tmp/tmpXXXXX_generated.py
🚀 Executing: python /tmp/tmpXXXXX_generated.py (timeout: 300s)
=== Executing Original Program ===
✅ Original execution successful
=== Executing Compiled Program fullgraph=False ===
✅ Compiled execution successful
=== Executing Compiled Program dynamic=True ===
✅ Compiled execution successful
✅ SUCCESS - artifacts saved to: /tmp/fuzzing_seed_42_1695123456789_success
```

## Generated Artifacts

Each test run creates artifacts in `/tmp/fuzzing_seed_{seed}_{timestamp}_{status}/`:

- **`summary.txt`** - Test run metadata
- **`operation_stack.txt`** - Human-readable operation sequence
- **`generated_code.py`** - Executable Python code
- **`operation_stack_diagram.png`** - Visual operation stack diagram
  ```
  Example diagram visualization:

  ┌─────────────────────────────────────────────────────────────┐
  │ Target: TensorSpec(size=(2,3), stride=(3,1), dtype=float32) │
  └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────┐
  │           torch.ops.aten.add (depth 0)                     │
  │     TensorSpec(size=(2,3), stride=(3,1), dtype=float32)    │
  └─────────────────────────────────────────────────────────────┘
                      │                         │
                      ▼                         ▼
  ┌─────────────────────────┐     ┌─────────────────────────────┐
  │      arg (depth 1)      │     │   torch.ops.aten.mul       │
  │ TensorSpec(size=(2,3))  │     │      (depth 1)             │
  │   dtype=float32         │     │ TensorSpec(size=(2,3))     │
  └─────────────────────────┘     │   dtype=float32            │
                                  └─────────────────────────────┘
                                              │                │
                                              ▼                ▼
                                  ┌─────────────────┐ ┌──────────────┐
                                  │  arg (depth 2)  │ │constant      │
                                  │ TensorSpec(...) │ │(depth 2)     │
                                  │  dtype=int32    │ │ScalarSpec    │
                                  └─────────────────┘ │dtype=float32 │
                                                      └──────────────┘
  ```

## Configuration

### Depth Control

Higher `max_depth` values generate more complex operation chains:
- `depth=1`: Simple operations (mostly leaf nodes)
- `depth=3`: Moderate complexity (default)
- `depth=10`: Very complex chains

### Known Issues Handling

TorchFuzz automatically skips known PyTorch issues or previously found, you should add them to the list known_issues.

```python
known_issues = {
    "RuntimeError: self.stride(-1) must be 1 to view ComplexDouble as":
        "https://github.com/pytorch/pytorch/issues/162561",
    "BooleanAtom not allowed in this context":
        "https://github.com/pytorch/pytorch/issues/160726",
}
```

## Debugging Failed Tests

When a test fails, examine the generated artifacts:

1. **Check `summary.txt`** for test parameters
2. **Review `generated_code.py`** for the exact failing code
3. **Examine `operation_stack.txt`** for the operation sequence
4. **View `operation_stack_diagram.png`** for visual understanding

## API Usage

### Programmatic Interface

```python
from fuzzer import fuzz_and_execute
from ops_fuzzer import fuzz_operation_stack, fuzz_spec
from codegen import convert_stack_to_python_code

# Generate and execute a single test
seed, success, error = fuzz_and_execute(seed=42, max_depth=3)

# Generate operation stack only
target_spec = fuzz_spec()
operation_stack = fuzz_operation_stack(target_spec, max_depth=3, seed=42)

# Generate code without executing
python_code = convert_stack_to_python_code(operation_stack, target_spec, seed=42)
```

## Testing Strategies

### Systematic Testing

```bash
# Test specific seed range
for i in {1000..1100}; do
    python fuzzer.py --single --seed $i --max-depth 5
done
```

### Continuous Integration

```bash
# Run 100 tests with timeout
timeout 300 python fuzzer.py --test --seed 42 --max-depth 3
```

### Regression Testing

```bash
# Test known problematic seeds
python fuzzer.py --single --seed 12345  # Known to trigger specific issue
```

## Contributing

### Adding New Operations

1. **Define operation in `ops_fuzzer.py`**:
   ```python
   def _get_new_op_args_specs(target_spec):
       return "new_op_name", [input_spec1, input_spec2]
   ```

2. **Add code generation in `codegen.py`**:
   ```python
   elif op_name == "new_op_name":
       return [f"{output_var} = torch.ops.aten.new_op({input_vars[0]}, {input_vars[1]})"]
   ```


3. **Update operation selection in `fuzz_op()`**
TODO: link an example PR that adds an operation.
