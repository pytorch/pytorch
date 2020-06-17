#include <torch/library.h>

#include <ATen/core/boxing/KernelFunction.h>

using torch::CppFunction;

TORCH_LIBRARY_IMPL(_, CPU,    m) { m.fallback(CppFunction::makeFallthrough()); }
TORCH_LIBRARY_IMPL(_, CUDA,   m) { m.fallback(CppFunction::makeFallthrough()); }
TORCH_LIBRARY_IMPL(_, HIP,    m) { m.fallback(CppFunction::makeFallthrough()); }
TORCH_LIBRARY_IMPL(_, FPGA,   m) { m.fallback(CppFunction::makeFallthrough()); }
TORCH_LIBRARY_IMPL(_, MSNPU,  m) { m.fallback(CppFunction::makeFallthrough()); }
TORCH_LIBRARY_IMPL(_, XLA,    m) { m.fallback(CppFunction::makeFallthrough()); }
TORCH_LIBRARY_IMPL(_, Vulkan, m) { m.fallback(CppFunction::makeFallthrough()); }

// For now, not including the more exotic backends like QuantizedCPU or
// ComplexCPU.  Maybe they should be included; if so, just add them to this
// list.
