//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/NativeFunctions.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/Copy.h>
#include <torch/library.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

using namespace at::mps;

namespace at {
namespace native {

Scalar _local_scalar_dense_mps(const Tensor& self) {
  Scalar r;

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_local_scalar_dense_mps", [&] {
        Tensor output = at::empty_like(self, kCPU);

        Tensor cpu_output = mps::mps_copy_(output, self, false);
        scalar_t value = *cpu_output.data_ptr<scalar_t>();
        r = Scalar(value);
   });

  return r;
}


} // namespace native
} // namespace at
