//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/_local_scalar_dense_native.h>

using namespace at::mps;

namespace at::native {

Scalar _local_scalar_dense_mps(const Tensor& self) {
  Scalar r;
  TORCH_CHECK(self.numel() > 0, "_local_scalar_dense: Empty tensor not supported");

  auto output = at::empty_like(self, TensorOptions(kCPU));
  mps::mps_copy_(output, self, false);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6(at::ScalarType::Half,
                                         at::ScalarType::Bool,
                                         at::ScalarType::BFloat16,
                                         at::ScalarType::UInt16,
                                         at::ScalarType::UInt32,
                                         at::ScalarType::UInt64,
                                         self.scalar_type(),
                                         "_local_scalar_dense_mps",
                                         [&] {
                                           scalar_t value = *output.data_ptr<scalar_t>();
                                           r = Scalar(value);
                                         });

  return r;
}

} // namespace at::native
