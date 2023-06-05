//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/mps/OperationUtils.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

using namespace at::mps;

namespace at::native {

Scalar _local_scalar_dense_mps(const Tensor& self) {
  Scalar r;

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half,
                                         at::ScalarType::Bool,
                                         at::ScalarType::BFloat16,
                                         self.scalar_type(),
                                         "_local_scalar_dense_mps",
                                         [&] {
                                           Tensor output = at::empty({1}, TensorOptions(at::CPU(self.scalar_type())));

                                           mps::mps_copy_(output, self, false);
                                           scalar_t value = *output.data_ptr<scalar_t>();
                                           r = Scalar(value);
                                         });

  return r;
}

} // namespace at::native
