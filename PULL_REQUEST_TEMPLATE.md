# Add GPU vs CPU Performance Benchmarking Utilities

## Summary

This PR adds comprehensive performance benchmarking utilities to PyTorch, enabling users to compare GPU and CPU performance for matrix operations and neural network training across different devices (CUDA, MPS, CPU).

## Motivation

PyTorch users frequently need to:
- Validate their PyTorch installation works correctly
- Understand GPU vs CPU performance trade-offs for their hardware
- Benchmark PyTorch operations for optimization decisions
- Verify GPU acceleration is working as expected

Currently, there's no standardized way to perform these benchmarks within PyTorch itself.

## Changes Made

### New Files Added:

1. **`benchmarks/performance_tests/gpu_cpu_benchmark.py`**
   - Main benchmarking utilities with modular class structure
   - Device detection (CUDA > MPS > CPU priority)
   - Matrix multiplication benchmarking with proper GPU synchronization
   - Neural network training benchmarking
   - Structured performance reporting

2. **`test/test_utils/test_performance_utils.py`**
   - Comprehensive unit tests for all benchmark utilities
   - Device-specific test cases (CPU, CUDA, MPS)
   - Integration tests for full benchmark runs
   - Edge case handling verification

3. **`tools/testing/run_performance_tests.sh`**
   - Automated test runner script
   - Environment detection and setup
   - Flexible test execution options

## Features

- ✅ **Cross-platform support**: Works with NVIDIA CUDA, Apple MPS, and CPU
- ✅ **Proper GPU synchronization**: Accurate timing with device synchronization
- ✅ **Modular design**: Easily extensible for additional benchmark types
- ✅ **Comprehensive testing**: Full unit test coverage
- ✅ **Type hints**: Complete type annotations for better IDE support
- ✅ **Documentation**: Detailed docstrings and usage examples

## Testing

### Unit Tests
```bash
python -m pytest test/test_utils/test_performance_utils.py -v
```

### Manual Testing
```bash
python benchmarks/performance_tests/gpu_cpu_benchmark.py
```

### Example Output
```
🚀 PyTorch GPU vs CPU Performance Benchmark
======================================================================
PyTorch version: 2.8.0

📊 Matrix Multiplication Results:
----------------------------------------------------------------------
Size     CPU (s)    GPU (s)    Speedup    Winner    
----------------------------------------------------------------------
2000x2000 0.0074    0.0043     1.70x      🚀 GPU
8000x8000 0.3598    0.1620     2.22x      🚀 GPU

🧠 Neural Network Training Performance
🚀 GPU is 2.14x faster for neural network training!
```

## Code Quality

- **Style**: Follows PEP 8 and PyTorch coding standards
- **Type Safety**: Complete type hints for all public APIs
- **Error Handling**: Graceful fallback for missing GPU support
- **Performance**: Minimal overhead, efficient memory usage
- **Modularity**: Clean separation of concerns with testable components

## Backwards Compatibility

✅ No breaking changes - all new functionality in separate modules

## Documentation

- Comprehensive docstrings with Args/Returns documentation
- Usage examples in `CONTRIBUTION_GUIDE.md`
- Clear error messages and helpful output formatting

## Benefits to PyTorch Community

1. **Installation Validation** - Users can verify PyTorch works correctly
2. **Hardware Optimization** - Understand GPU vs CPU performance characteristics  
3. **Educational Value** - Demonstrates proper GPU benchmarking techniques
4. **Regression Testing** - Detect performance regressions in PyTorch releases
5. **Cross-Platform** - Works consistently across different hardware

## Future Enhancements

This foundation enables future improvements like:
- Integration with TorchBench
- Additional operation types (convolutions, attention mechanisms)
- Memory usage profiling
- Multi-GPU benchmarking

## Checklist

- [x] Code follows PyTorch style guidelines
- [x] Unit tests added with good coverage
- [x] Documentation and docstrings included
- [x] No breaking changes to existing functionality
- [x] Cross-platform compatibility verified
- [x] Type hints provided for all public APIs
- [x] Error handling implemented
- [x] Performance impact is minimal

## Related Issues

Addresses community requests for standardized PyTorch performance benchmarking tools.

---

**Note**: This contribution provides a solid foundation for PyTorch performance benchmarking that can be extended by the community over time.
