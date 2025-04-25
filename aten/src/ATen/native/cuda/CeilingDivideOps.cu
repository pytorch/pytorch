#include <c10/core/Device.h>

// Forward declarations to avoid all headers
namespace at {
class Tensor;
namespace native {
TORCH_API Tensor ceiling_divide_cpu(const Tensor& self, const Tensor& other);
TORCH_API Tensor& ceiling_divide__cpu(Tensor& self, const Tensor& other);
TORCH_API Tensor& ceiling_divide_out_cpu(const Tensor& self, const Tensor& other, Tensor& result);
}
}

// This file only defines the CUDA-specific implementations
namespace at::native {

// Simple CUDA stubs that just call the CPU versions
Tensor ceiling_divide(const Tensor& self, const Tensor& other) {
  // Just delegate to CPU implementation
  return ceiling_divide_cpu(self, other);
}

Tensor& ceiling_divide_(Tensor& self, const Tensor& other) {
  // Just delegate to CPU implementation
  return ceiling_divide__cpu(self, other);
}

Tensor& ceiling_divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  // Just delegate to CPU implementation
  return ceiling_divide_out_cpu(self, other, result);
}

} // namespace at::native 