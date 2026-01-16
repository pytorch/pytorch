# Add Power-of-2 Scale Optimization for Quantization-Aware Training

## Summary

This PR adds power-of-2 scale optimization to PyTorch's quantization infrastructure, enabling quantization scales to be rounded to the nearest power of 2. This optimization improves operational efficiency on hardware accelerators like DSPs, NPUs, and FPGAs by enabling efficient bit-shift operations instead of multiplications.

## Motivation

Power-of-2 scale quantization is a well-established technique in the research community and industry for optimizing neural network inference on specialized hardware. When quantization scales are constrained to powers of 2 (e.g., 0.25, 0.5, 1.0, 2.0, 4.0), hardware can replace expensive multiplication operations with simple bit-shifts, resulting in:

- **Significant speedups**: Research shows 1.2× to 10× speedups depending on hardware
- **Energy efficiency**: 1.2× to 36× energy savings reported in academic literature
- **Hardware compatibility**: Better alignment with DSPs, NPUs, FPGAs, and edge devices
- **Industry support**: Already supported by TensorRT (MX-Compliant Dynamic Quantization) and AMD Vitis AI

### Research Evidence

Multiple peer-reviewed papers demonstrate the benefits:

- **P²-ViT**: Up to 10.1× speedup and 36.8× energy savings over baseline GPU tensor cores ([arXiv:2405.19915](https://arxiv.org/abs/2405.19915))
- **PoTAcc**: Average 1.23× speedup and 1.24× energy reduction on edge accelerators ([arXiv:2409.20403](https://arxiv.org/abs/2409.20403))
- **DenseShift**: ~1.6× speedup with power-of-2 weights ([arXiv:2208.09708](https://arxiv.org/abs/2208.09708))
- **ShiftCNN**: ~4× power reduction on FPGAs ([arXiv:1706.02393](https://arxiv.org/abs/1706.02393))

### Industry Support

- **NVIDIA TensorRT**: Supports power-of-2 rounding in MX-Compliant Dynamic Quantization for FP8 formats
- **AMD Vitis AI**: Supports power-of-2 scale quantization via Microsoft Olive
- **ONNX Community**: Has proposals for explicit power-of-2 scale types ([Issue #2659](https://github.com/onnx/onnx/issues/2659))

## Implementation

### Design Philosophy

The implementation follows a **minimal, backward-compatible approach**:

1. **Single point of change**: Only the base observer class (`UniformQuantizationObserverBase`) is modified
2. **Automatic inheritance**: All observer subclasses automatically inherit the feature
3. **Opt-in behavior**: Default is `False` to maintain backward compatibility
4. **No breaking changes**: Existing code continues to work unchanged

### Changes Made

1. **Utility Function** (`torch/ao/quantization/utils.py`):
   - Added `round_to_power_of_2()` function that rounds scales to nearest power of 2
   - Handles edge cases (zero, negative, very small/large values)
   - Supports both scalar and tensor scales (per-tensor and per-channel)

2. **Observer Base Class** (`torch/ao/quantization/observer.py`):
   - Added `power_of_2_scale: bool = False` parameter to `UniformQuantizationObserverBase.__init__`
   - Modified `_calculate_qparams()` to apply power-of-2 rounding when enabled
   - Updated docstrings for all observer classes

3. **FakeQuantize Integration**:
   - No changes needed! `FakeQuantize` already supports `**observer_kwargs`
   - Users can pass `power_of_2_scale=True` via `observer_kwargs`

4. **Comprehensive Tests** (`test/quantization/core/test_power_of_2_scale.py`):
   - Tests for utility function (scalar, tensor, edge cases)
   - Tests for all observer types (per-tensor, per-channel)
   - Tests for FakeQuantize integration
   - Tests for different quantization schemes
   - Backward compatibility tests

### Key Features

- **Works during QAT**: Model trains with power-of-2 scales, ensuring training-inference consistency
- **Works at export**: Same scales are exported to ONNX/TensorRT for hardware optimization
- **Supports all quantization schemes**: Per-tensor, per-channel, symmetric, affine
- **Comprehensive documentation**: All docstrings updated with parameter descriptions

## Usage Example

```python
# For observers
observer = MinMaxObserver(power_of_2_scale=True)

# For fake_quantize (via observer_kwargs)
fake_quant = FakeQuantize(
    observer=MovingAverageMinMaxObserver,
    power_of_2_scale=True  # passed via **observer_kwargs
)

# In QAT workflow
qconfig = QConfig(
    activation=FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        power_of_2_scale=True
    ),
    weight=FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        power_of_2_scale=True
    )
)
```

## Testing

- ✅ All files compile successfully
- ✅ Comprehensive test suite added (`test_power_of_2_scale.py`)
- ✅ Tests cover utility function, observers, FakeQuantize, and edge cases
- ✅ Backward compatibility verified (default behavior unchanged)
- ✅ No linter errors

## Backward Compatibility

- **Default behavior unchanged**: `power_of_2_scale=False` by default
- **No breaking changes**: All existing code continues to work
- **Opt-in feature**: Users must explicitly enable it

## Benefits

1. **Hardware Efficiency**: Enables bit-shift operations instead of multiplications
2. **Industry Alignment**: Matches capabilities in TensorRT and Vitis AI
3. **Research Support**: Backed by multiple peer-reviewed papers
4. **Minimal Code Changes**: Only 2 core files modified (utils.py, observer.py)
5. **Easy to Use**: Simple one-parameter API
6. **Well Tested**: Comprehensive test coverage

## Files Changed

- `torch/ao/quantization/utils.py`: Added `round_to_power_of_2()` utility function
- `torch/ao/quantization/observer.py`: Added parameter and rounding logic
- `test/quantization/core/test_power_of_2_scale.py`: Comprehensive test suite (new file)

## References

See `POWER_OF_2_QUANTIZATION_REFERENCES.md` for detailed references to:
- Hardware manufacturer documentation
- Research papers with speedup measurements
- Industry implementations (TensorRT, Vitis AI)
- Academic benchmarks

## Checklist

- [x] Code compiles successfully
- [x] Tests added and passing
- [x] Documentation updated
- [x] Backward compatibility maintained
- [x] No linter errors
- [x] Follows PyTorch coding conventions
