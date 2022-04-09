#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ReduceAllOps.h>

#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_aminmax_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/max_native.h>
#include <ATen/ops/min_native.h>
#endif

namespace at {
namespace native {

DEFINE_DISPATCH(min_all_stub);
DEFINE_DISPATCH(max_all_stub);

Tensor min(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0,
              "min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  Tensor result = at::empty({}, self.options());
  min_all_stub(self.device().type(), result, self.contiguous());
  return result;
}

Tensor max(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0,
              "max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  Tensor result = at::empty({}, self.options());
  max_all_stub(self.device().type(), result, self.contiguous());
  return result;
}

// DEPRECATED: Use at::aminmax instead
std::tuple<Tensor, Tensor> _aminmax_all(const Tensor &self) {
  TORCH_WARN_ONCE("_aminmax is deprecated as of PyTorch 1.11 and will be removed in a future release. Use aminmax instead."
                  " This warning will only appear once per process.");
  return at::aminmax(self);
}

}} // namespace at::native
