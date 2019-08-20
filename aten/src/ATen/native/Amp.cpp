#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Amp.h>
#include <ATen/native/TensorIterator.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <vector>
#include <unordered_map>

namespace at {
namespace native {

// extern may be unnecessary here, it's just a declaration.
extern Tensor _amp_overflow_state_cuda(const Tensor &);

Tensor _amp_overflow_state(const Tensor & new_state) {
  return at::native::_amp_overflow_state_cuda(new_state);
}

DEFINE_DISPATCH(amp_unscale_inf_check_stub);

// Entry point for torch binding, as specified in native_functions.yaml
// scale must be an external argument, so it can be saved for backward.
Tensor _amp_unscale_inf_check_cuda(const Tensor & scaled_grad, double scale) {
  TORCH_CHECK(scaled_grad.is_cuda());

  auto unscaled_grad = at::empty_like(scaled_grad);

  auto iter = TensorIterator::unary_op(unscaled_grad, scaled_grad);

  amp_unscale_inf_check_stub(kCUDA, iter, scale);

  return iter.output();
}

} // namespace native
} // namespace at
