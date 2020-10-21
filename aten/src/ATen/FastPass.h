#include <ATen/ATen.h>

namespace at {
TORCH_API Tensor scalar_tensor_fast(Scalar s, const TensorOptions& options);
} //namespace at
