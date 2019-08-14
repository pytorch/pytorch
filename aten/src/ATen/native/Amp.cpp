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
namespace {
  // Should these be thread local, or global and mutexed?
  std::unordered_map<TensorImpl*, Tensor&> cached_leaf_casts;
  // Alternative, if we want to make absolutely sure the cached Tensors stay alive:
  // std::unordered_map<void*, Tensor> fp16_casted_leaves;
}

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

// Tensor _amp_cached_cast(const Tensor & input,
//                        ScalarType input_type, // input_type is passed as a separate arg so it can be stashed for backward.
//                        ScalarType output_type)
// {
//   TORCH_CHECK(input.is_cuda(), "input must be cuda");
//   TORCH_CHECK(input.scalar_type() == input_type, "input.scalar_type() must be equal to input_type");
//   TORCH_CHECK(input.is_leaf() == input_is_leaf, "input.is_leaf() must be equal to input_is_leaf");
//
//   if(output_type != input_type)
//   {
//     bool can_try_cache = (input.is_leaf() && input.scalar_type() == kFloat);
//     if(can_try_cache)
//     {
//       auto it = cached_leaf_casts.find(input.unsafeGetTensorImpl()); // Use the owned TensorImpl* as a Tensor's uuid.
//       if(it != cached_leaf_casts.end())
//         return it->second; // Return the cached value.
//       else
//       {
//         auto casted_input = input.to(kHalf);
//         cached_leaf_casts.emplace(input.unsafeGetTensorImpl(), casted_input);
//         return casted_input;
//       }
//     }
//     else
//       return input.to(output_type);
//   }
//
//   return input;
// }
//
//
// Tensor _amp_cached_cast_backward(const Tensor & grad_output,
//                                  ScalarType input_was_type,
//                                  ScalarType output_was_type)
// {
//   return grad_output.to(input_was_type);
// }

} // namespace native
} // namespace at
