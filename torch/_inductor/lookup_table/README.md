# Template Lookup Table System

The template lookup table system provides a way to pre-configure kernel template parameters for specific operations and
input configurations, bypassing the default choice generation and autotuning process.

## Overview

The lookup table system replaces default choice generation with pre-configured template parameters for specific
operations and input configurations. It sits orthogonal to `max-autotune(-gemm)` in the following way

If a lookup table is provided and there is a match

- We check whether the template(s) in the match are currently in use
- If so, we use the pre-configured template(s) and config and bypass choice generation
  - If more than one choice is provided, we run autotune among the pre-configured choices
- If not, we fall back to the default choice generation process, including max-autotune(-gemm) logic

If there is no match, we fall back to the default choice generation process, including max-autotune(-gemm) logic

## Configuration

Enable the system by setting both:

```python
from torch._inductor import config
config.lookup_table.table = your_table_dict
# You also need to set it as the default choice handler
from torch._inductor.lookup_table import LookupTableChoices
torch._inductor.V.set_choices_handler(LookupTableChoices())
```

### Device Key Handling

The key schema format is described in detail in the [Key Schemas](#key-schemas) section below.

Configure device key behavior:

```python
# Control whether entries include device-specific keys for lookups
# Device-agnostic entries work across different GPU models
```

**Lookup Behavior**: During lookup, the system automatically tries both key formats:

1. **Device-specific key** (e.g., `"NVIDIA H100+input_data+mm"`) - tried first
1. **Device-agnostic key** (e.g., `"input_data+mm"`) - tried if device-specific fails

**Priority**: If both device-specific and device-agnostic entries exist for the same inputs, the device-specific entry
takes priority.

**NOTE**: Device-based keys simplify hardware-specific optimization without complex build rules. Currently limited to
device name only. If you need additional conditional key attributes (e.g., CUDA version filtering), please file an issue
or submit a patch.

## Behavior

When the table is active, the following behavior occurs for all supported operations:

### Match Found

- Uses pre-configured choices from the table instead of generating default choices
- Bypasses autotuning if only a single choice is provided
- If multiple choices are provided, autotuning occurs among those choices only

### No Match Found

- Standard default behavior - generates choices using heuristics and max-autotune settings

### Table Not Set or Inactive

- Standard default behavior - generates choices using heuristics and max-autotune settings

## Supported Operations

Currently supports: `mm`, `addmm`, `bmm`, `mm_plus_mm`, `scaled_mm` operations with

- Triton
- ATEN
- DecomposeK

## Table Format

The table is a dictionary with keys in the format:

```
"input_key+op_name"
```

Where:

- `input_key`: Generated from `KernelInputs.key` property, represents tensor shapes/dtypes/strides
- `op_name`: Operation name (`"mm"`, `"addmm"`, etc.)

Each value is a list of configuration dictionaries containing:

- `template_id`: Template identifier (`"triton:mm"`, `"triton::mm_persistent_tma"`, `"decompose_k"`, etc.)
- Template-specific parameters (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, etc.)

## Key Schemas

**NOTE**: The key schema format is subject to change as the system evolves.

The lookup table uses composite keys to match kernel configurations. See
[Implementation Details](#implementation-details) below for more technical information about key generation. This
section describes the structure of these keys.

### Key Format Structure

Keys follow the pattern:

```
[device_name+]input_key+[additional_params+]op_name
```

Components:

- **device_name** (optional): GPU device identifier (e.g., `"NVIDIA H100"`)

  - Obtained from `torch.cuda.get_device_properties().gcnArchName`
  - Enables device-specific optimizations
  - When omitted, creates device-agnostic entries that work across hardware

- **input_key**: Tensor configuration representation from `KernelInputs.key`

  - Format: `((dtype, shape, stride), (dtype, shape, stride), ...)`
  - Each tuple represents one input tensor's properties
  - Example: `((torch.float16, [128, 256], [0, 1]), (torch.float16, [64, 256], [256, 1]))`
  - Order matches the operation's input argument order

- **additional_params** (optional): Operation-specific parameters

  - Format: `key1=value1&key2=value2`
  - Example: `alpha=1&beta=1` for addmm operations

- **op_name**: Operation identifier

  - Examples: `"mm"`, `"addmm"`, `"bmm"`, `"mm_plus_mm"`, `"scaled_mm"`

### Key Examples

**Device-specific key for addmm:**

```
"NVIDIA H100+((torch.float16, [128, 256], [0, 1]), (torch.float16, [128, 64], [64, 1]), (torch.float16, [64, 256], [256, 1]))+alpha=1&beta=1+addmm"
```

**Device-agnostic key for mm:**

```
"((torch.float16, [64, 128], [128, 1]), (torch.float16, [128, 256], [256, 1]))+mm"
```

**Key with no additional parameters:**

```
"((torch.float32, [512, 512], [512, 1]), (torch.float32, [512, 512], [512, 1]))+bmm"
```

### Lookup Strategy

During lookup, the system tries keys in priority order:

1. **Device-specific key** - checked first if device information is available
1. **Device-agnostic key** - fallback if device-specific lookup fails

This allows tables to contain:

- Device-optimized configurations (higher priority)
- Portable configurations that work across devices
- Mix of both for flexible deployment

## Example Table

This is an example table for a single input showing two configurations

```python
table = {
  "((torch.float16, [128, 256], [0, 1]), (torch.float16, [128, 64], [64, 1]), (torch.float16, [64, 256], [256, 1]))+alpha=1&beta=1+addmm": [
    {
      "template_id": "triton::mm",
      "EVEN_K": true,
      "USE_FAST_ACCUM": false,
      "ACC_TYPE": "tl.float32",
      "num_stages": 2,
      "num_warps": 4,
      "BLOCK_M": 32,
      "BLOCK_N": 32,
      "BLOCK_K": 64,
      "hint_override": null,
      "GROUP_M": 8,
      "template_hash": "0717af5834e39dcca7ea817f896b8d85b4886422da7a3ab5f6911b4cfe568896"
    },
    {
      "template_id": "aten::bias_addmm"
    },
  ]
}
```

## Source Hashing Safety

The lookup table system includes source hashing to prevent using stale configurations when template code changes.

### Configuration

- **Enabled by default**: `torch._inductor.config.lookup_table.check_src_hash = True`
- **Optional field**: Add `"template_hash"` to table entries for enhanced safety

### Behavior

When source hash checking is enabled:

- Template configurations with `"template_hash"` fields are validated against current template source hashes
- Mismatched hashes indicate the template code has changed since the configuration was created
- Stale configurations are automatically filtered out with a warning message
- Configurations without hash fields are preserved for backward compatibility or if the user wants to fly looser

### Example with Template Hash

```python
{
  "template_id": "triton::mm",
  "BLOCK_M": 32,
  "BLOCK_N": 32,
  "BLOCK_K": 16,
  "template_hash": "0717af5834e39dcca7ea817f896b8d85b4886422da7a3ab5f6911b4cfe568896"
}
```

## Performance Impact

- **Lookup Hit**: Eliminates heuristic choice generation and autotuning overhead (if a single choice)
- **Lookup Miss**: Default behavior, including heuristic choice generation and autotuning
- **Memory**: Table stored in memory, minimal overhead for key generation and lookup

## Implementation Details

### Key Generation

- Device key: Uses `torch.cuda.get_device_properties().gcnArchName` (e.g., "NVIDIA H100")
- Input key: Generated from `KernelInputs.key` containing tensor properties

### Entry Points

The system is accessed through:

- `lookup_template_configs(kernel_inputs, op_name, template_uids)` - Main lookup function
- `LookupTableChoices._finalize_template_configs()` - Integration point with existing choice system

### Error Handling

- Validates config dictionaries contain required `template_id` field
- Gracefully handles non-CUDA devices by returning empty results
<<<<<<< HEAD
=======

## Recording Lookup Tables

The system can record autotuning results to automatically build lookup tables for future use. This eliminates the need to manually create table entries and ensures optimal configurations are captured from real workloads.

### Quick Start: Recording

1. **Enable recording:**
   ```python
   from torch._inductor import config

   # Master switch - must be True to enable recording
   config.lookup_table.recording_active = True

   # Configure recording behavior
   config.lookup_table.recorder_topk = 5  # Record top 5 fastest choices per operation
   config.lookup_table.recorder_record_dir = "/path/to/output"  # Save to files
   ```

2. **Run your model with autotuning:**
   ```python
   import torch

   model = YourModel()
   compiled_model = torch.compile(model, mode="max-autotune")

   # Recording happens automatically during compilation and execution
   result = compiled_model(inputs)
   ```

3. **Files are automatically saved** to the specified directory with timestamped filenames like `inductor_lut_20241205_143052_123.json`.

### Recording Configuration

All recording options are available under `torch._inductor.config.lookup_table`:

```python
from torch._inductor import config

# Master recording switch - must be True for any recording to happen
config.lookup_table.recording_active = True  # Default: False

# Logging and immediate emission
config.lookup_table.recorder_emit = True  # Default: True (logs entries)

# File recording - set directory to enable file output
config.lookup_table.recorder_record_dir = "/path/to/save/tables"  # Default: None

# Number of top choices to record per operation key
config.lookup_table.recorder_topk = 10  # Default: None (record all)
config.lookup_table.recorder_topk = 0   # Special case: disable recording

# Template safety and portability options
config.lookup_table.record_template_hash = True   # Default: True (include hashes)
config.lookup_table.record_with_device_key = True # Default: True (device-specific keys)
```

### Understanding TopK: Determinism vs. Flexibility

The `recorder_topk` setting is crucial for controlling the behavior of your recorded lookup tables:

#### TopK = 1: Maximum Performance and Determinism
```python
config.lookup_table.recorder_topk = 1  # Record only the fastest choice
```

**Benefits:**
- **No autotuning overhead**: When using the recorded table, exactly one choice is available, so no autotuning occurs
- **Perfect determinism**: Always uses the same kernel for identical inputs across runs
- **Fastest compilation**: Minimal overhead during `torch.compile()` with the lookup table
- **Production-ready**: Ideal for deployment where consistency and speed matter most

**Use case:** Production environments where you want maximum performance and deterministic behavior.

#### TopK > 1: Balanced Performance with Options
```python
config.lookup_table.recorder_topk = 5  # Record top 5 fastest choices
```

**Benefits:**
- **Some autotuning**: When using the recorded table, autotuning occurs among the recorded choices (faster than full autotuning)
- **Flexibility**: Multiple good options available if hardware characteristics change slightly
- **Robustness**: Backup choices if the fastest choice becomes unavailable

**Trade-offs:**
- **Slight overhead**: Autotuning still occurs among the recorded choices
- **Less determinism**: May pick different choices between runs based on timing variations

**Use case:** Development/staging environments where you want good performance but retain some flexibility.

#### TopK = None: Maximum Visibility for Analysis
```python
config.lookup_table.recorder_topk = None  # Record ALL choices that were tested
```

**Benefits:**
- **Complete picture**: See every template choice that was considered during autotuning
- **Manual optimization**: Analyze all options and manually edit the table to select specific choices
- **Debugging**: Understand what choices were available and their relative performance

**Trade-offs:**
- **Large tables**: More storage space and memory usage
- **Full autotuning**: When using the table, autotuning occurs among all recorded choices (no speed benefit)

**Use case:** Analysis, debugging, or when you want to manually curate the final lookup table.

#### Recommended Strategy

1. **Start with TopK = None** for analysis:
   ```python
   config.lookup_table.recorder_topk = None  # See all options
   ```

2. **Analyze the results** to understand choice distribution and performance gaps

3. **Switch to TopK = 1** for production:
   ```python
   config.lookup_table.recorder_topk = 1  # Lock in the fastest choice
   ```

4. **Validate determinism** by running the same workload multiple times and confirming identical kernels

### Device Key Configuration for Recording

The `record_with_device_key` setting controls whether recorded entries are device-specific or portable:

```python
# Device-specific recording (more precise but less portable)
config.lookup_table.record_with_device_key = True
# Key format: "NVIDIA H100+input_shapes+operation"
# Best for: Production environments with known hardware

# Device-agnostic recording (more portable across GPU types)
config.lookup_table.record_with_device_key = False
# Key format: "input_shapes+operation"
# Best for: Development, CI/CD, mixed GPU environments
```

**Note**: During lookup, both key formats are always tried regardless of this setting, with device-specific keys taking priority if both exist.

### How Recording Works

The recording system automatically:

1. **Captures autotuning results**: Monitors all kernel selection decisions during `torch.compile()` execution
2. **Filters by performance**: Records only the fastest choices (configurable via `recorder_topk`)
3. **Generates lookup keys**: Uses the same key format as the lookup system for consistency
4. **Saves incrementally**: Each autotuning session appends to timestamped JSON files
5. **Maintains safety**: Includes template hashes to prevent using stale configurations

### Example: Complete Recording Workflow

```python
import torch
from torch._inductor import config
import tempfile
import json

# Enable recording with configuration
config.lookup_table.recording_active = True
config.lookup_table.recorder_topk = 3
config.lookup_table.record_template_hash = True

# Use temporary directory for this example
with tempfile.TemporaryDirectory() as temp_dir:
    config.lookup_table.recorder_record_dir = temp_dir

    # Your model
    def matmul_model(a, b):
        return torch.mm(a, b)

    # Compile with autotuning (triggers recording)
    compiled_model = torch.compile(matmul_model, mode="max-autotune")

    # Run the model (autotuning results are recorded automatically)
    a = torch.randn(512, 512, device='cuda', dtype=torch.float16)
    b = torch.randn(512, 512, device='cuda', dtype=torch.float16)
    result = compiled_model(a, b)

    # Check recorded files
    import os
    files = [f for f in os.listdir(temp_dir) if f.startswith('inductor_lut_')]
    print(f"Recorded files: {files}")

    # Load and inspect the recorded lookup table
    if files:
        with open(os.path.join(temp_dir, files[0])) as f:
            recorded_table = json.load(f)

        print("Recorded entries:")
        for key, configs in recorded_table.items():
            print(f"  Key: {key}")
            for i, config in enumerate(configs):
                print(f"    Config {i+1}: template_id={config['template_id']}")
```

### Using Recorded Tables

Once you have recorded tables, use them for faster compilation:

```python
from torch._inductor import config
from torch._inductor.lookup_table import LookupTableChoices
from torch._inductor.virtualized import V
import json

# Load your recorded table
with open('inductor_lut_20241205_143052_123.json') as f:
    lookup_table = json.load(f)

# Configure the system to use the lookup table
config.lookup_table.table = lookup_table
V.set_choices_handler(LookupTableChoices())

# Now compilation will use your recorded configurations
model = torch.compile(your_model, mode="max-autotune")
result = model(inputs)  # Uses lookup table, skips autotuning
```

### Advanced: Custom Recording Backends

Extend the recording system with custom backends for specialized workflows:

```python
from torch._inductor.lookup_table import recorder

# Custom emit backend for immediate processing
class CustomLogBackend(recorder.EmitBackend):
    def emit(self, entry):
        # Process each entry immediately as it's recorded
        print(f"Recorded {entry.key} -> {entry.value['template_id']} (runtime: {entry.runtime:.4f}ms)")

# Custom record backend for batch processing
class DatabaseRecordBackend(recorder.RecordBackend):
    def __init__(self, connection_string):
        self.conn = connection_string

    def dump(self, data):
        # Save all entries to database when dump() is called
        for key, entries in data.items():
            for entry in entries:
                self.save_to_db(key, entry.value, entry.runtime)

# Register custom backends
recorder.add_backend(CustomLogBackend())
recorder.add_backend(DatabaseRecordBackend("postgresql://..."))
```

### Performance and Overhead

**Recording Performance**:
- **Minimal overhead**: Recording adds ~1-5μs per kernel selection
- **Fast bail**: When `recording_active=False`, overhead is ~100ns (single boolean check)
- **Memory efficient**: Only keeps configured `topk` entries per operation in memory

**Storage**:
- **Typical size**: 1-10KB per recorded table file
- **Compression**: JSON format is human-readable and compresses well
- **Incremental**: Each compilation session creates a separate timestamped file

### Troubleshooting Recording

**No files created?**
```python
# Check if recording is properly enabled
from torch._inductor import config
print(f"Recording active: {config.lookup_table.recording_active}")
print(f"Record directory: {config.lookup_table.recorder_record_dir}")
print(f"TopK setting: {config.lookup_table.recorder_topk}")

# Ensure max-autotune is enabled to trigger template selection
compiled_model = torch.compile(model, mode="max-autotune")
```

**Empty tables?**
- Recording only captures results from operations that undergo autotuning
- Ensure your model has matrix operations (`mm`, `addmm`, `bmm`, etc.)
- Check that input sizes are large enough to trigger template-based kernels
- Verify GPU kernels are being used (CPU operations aren't recorded)

**Files but no entries?**
- Check `recorder_topk` isn't set to 0 (which disables recording)
- Ensure autotuning found valid template choices (not just ATEN fallbacks)
- Verify templates have the required `template_id` and parameters
>>>>>>> 8e9a06853941 ([inductor][lookup table] add recorder 2/3)
