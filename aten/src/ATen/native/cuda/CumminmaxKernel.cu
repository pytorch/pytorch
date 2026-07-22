#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>

#include <ATen/native/cuda/ScanKernels.h>
#include <ATen/native/cuda/ScanUtils.cuh>

#include <limits>
#include <functional>

namespace at::native {

void launch_cummax_cuda_kernel(const TensorBase& self, const TensorBase& values, const TensorBase& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16,
    self.scalar_type(), "cummax_cuda", [&]() {
    scalar_t init = self.is_floating_point() ? (-1*std::numeric_limits<scalar_t>::infinity()) : std::numeric_limits<scalar_t>::lowest();
    scan_dim_with_indices<scalar_t>(self, values, indices, dim, init, std::greater_equal<scalar_t>());
  });
}

void launch_cummin_cuda_kernel(const TensorBase& self, const TensorBase& values, const TensorBase& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16,
    self.scalar_type(), "cummin_cuda", [&]() {
    scalar_t init = self.is_floating_point() ? std::numeric_limits<scalar_t>::infinity() : std::numeric_limits<scalar_t>::max();
    scan_dim_with_indices<scalar_t>(self, values, indices, dim, init, std::less_equal<scalar_t>());
  });
}

} // namespace at::native
