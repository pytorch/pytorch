# Template Lookup Table System

The template lookup table system provides a way to pre-configure kernel template parameters for specific operations and input configurations, bypassing the default choice generation and autotuning process.

## Overview

The lookup table system replaces default choice generation with pre-configured template parameters for specific operations and input configurations. It sits orthogonal to `max-autotune(-gemm)` in the following way

If a lookup table is provided and there is a match
- We check whether the template(s) in the match are currently in use
- If so, we use the pre-configured template(s) and config and bypass choice generation
  - If more than one choice is provided, we run autotune among the pre-configured choices
- If not, we fall back to the default choice generation process, including max-autotune(-gemm) logic

If there is no match, we fall back to the default choice generation process, including max-autotune(-gemm) logic

## Configuration

Enable the system by setting both:
```python
torch._inductor.config.template_config_lookup_table.table = your_table_dict
# You also need to set it as the default choice handler
torch._inductor.V.set_choice_handler(torch._inductor.lookup_table.LookupTableChoices())
```

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
"input_key+op_name+tf32={bool}"
```

Where:
- `input_key`: Generated from `KernelInputs.key` property, represents tensor shapes/dtypes/strides
- `op_name`: Operation name (`"mm"`, `"addmm"`, etc.)
- `tf32`: Current TF32 setting (`torch.backends.cuda.matmul.allow_tf32`)

Each value is a list of configuration dictionaries containing:
- `template_id`: Template identifier (`"triton:mm"`, `"triton::mm_persistent_tma"`, `"decompose_k"`, etc.)
- Template-specific parameters (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, etc.)

### Example Table

This is an example table for a single input showing two configurations

```python
table = {
  "NVIDIA H100+((torch.float16, [128, 256], [0, 1]), (torch.float16, [128, 64], [64, 1]), (torch.float16, [64, 256], [256, 1]))+alpha=1&beta=1+addmm+tf32=False": [
    {
      "template_id": "triton::mm",
      "EVEN_K": true,
      "ALLOW_TF32": false,
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
- **Enabled by default**: `torch._inductor.config.template_config_lookup_table.check_src_hash = True`
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

## Recording Lookup Tables

The system can record autotuning results to build lookup tables for future use.

### How to Record

1. **Set recording directory:**
   ```bash
   export TORCH_INDUCTOR_LOOKUP_TABLE_RECORD_DIR="/path/to/output"
   ```

2. **Configure recording parameters:**
   ```python
   config_patch = {
       "template_config_lookup_table.recorder_topk": 10,  # Record top 10 choices
   }
   with torch._inductor.config.patch(config_patch):
       # Run your model with torch.compile
       compiled_model = torch.compile(model, mode="max-autotune")
       result = compiled_model(inputs)
   ```

3. **Dump recorded data:**
   ```python
   from torch._inductor.lookup_table import recorder
   recorder.dump()  # Saves timestamped JSON files
   ```

### Adding Custom Recorders

Extend the recording system by creating custom backends:

A EmitBackend is responsible for processing a single entry. A RecordBackend is
responsible for accumulating entries and processing them all at once when `dump()` is called.

```python
from torch._inductor.lookup_table.recorder import EmitBackend, RecordBackend, emit_backend, record_backend

@emit_backend()  # For immediate logging/processing
class CustomEmitBackend(EmitBackend):
    def emit(self, entry):
        # Process entry immediately
        pass

@record_backend(custom_param="value")  # For batch saving
class CustomRecordBackend(RecordBackend):
    def __init__(self, custom_param):
        self.param = custom_param

    def dump(self, data):
        # Save accumulated data
        pass
```

## Performance Impact

- **Lookup Hit**: Eliminates heuristic choice generation and autotuning overhead (if a single choice)
- **Lookup Miss**: Default behavior, including heuristic choice generation and autotuning
- **Memory**: Table stored in memory, minimal overhead for key generation and lookup

## Implementation Details

### Key Generation
- Device key: Uses `torch.cuda.get_device_properties().gcnArchName` (e.g., "NVIDIA H100")
- Input key: Generated from `KernelInputs.key` containing tensor properties
- Suffix: Always includes current TF32 setting for consistency

### Entry Points

The system is accessed through:
- `lookup_template_configs(kernel_inputs, op_name, template_uids)` - Main lookup function
- `LookupTableChoices._adjust_mm_configs()` - Integration point with existing choice system

### Error Handling
- Validates config dictionaries contain required `template_id` field
- Warns when filtering TF32 configs due to settings mismatch
- Gracefully handles non-CUDA devices by returning empty results
