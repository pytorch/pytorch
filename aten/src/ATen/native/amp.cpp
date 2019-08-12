#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <vector>
#include <unordered_map>

namespace at {
namespace native {
namespace {
  // Should these be thread local, or global and mutexed?
  float loss_scale;
  std::unordered_map<TensorImpl*, Tensor&> cached_leaf_casts;
  // Alternative, if we want to make absolutely sure the cached Tensors stay alive:
  // std::unordered_map<void*, Tensor> fp16_casted_leaves;
  at::Tensor amp_overflow_state;
  // This could avoid the use of is_variable(true) in the TensorOptions below, but apparently torch:: is not declared here
  // torch::Tensor amp_overflow_state;
}

Tensor _amp_overflow_state(const Tensor & new_state) {
  if(new_state.defined())
  {
    TORCH_CHECK(new_state.is_cuda(), "Overflow state must be a CUDA tensor.");
    TORCH_CHECK(new_state.numel() == 1, "Overflow state must be a 1-element tensor.");
    TORCH_CHECK(new_state.scalar_type() == at::ScalarType::Int, "Overflow state must be an int tensor.");
    amp_overflow_state = new_state;
  } else if(!amp_overflow_state.defined())
    amp_overflow_state = at::zeros({1}, at::device(kCUDA).dtype(kInt).is_variable(true));

  return amp_overflow_state;
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
