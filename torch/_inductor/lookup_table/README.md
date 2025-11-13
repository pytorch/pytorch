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
