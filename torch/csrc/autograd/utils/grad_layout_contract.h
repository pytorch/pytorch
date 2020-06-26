#pragma once

#include <ATen/Tensor.h>

namespace torch {
namespace autograd {
namespace utils {

// Helper functions to enforce the "Gradient Layout Contract" described in
// torch/csrc/autograd/AccumulateGrad.h.

// Checks if grad obeys the contract with variable.
inline bool obeys_layout_contract(const at::Tensor& grad, const at::Tensor& variable) {
  TORCH_INTERNAL_ASSERT(!grad.is_sparse());
  TORCH_INTERNAL_ASSERT(!variable.is_sparse());
  return variable.is_non_overlapping_and_dense() ?
         (grad.strides() == variable.strides()) :
         grad.is_contiguous(at::MemoryFormat::Contiguous);
}

// Creates a clone of new_grad that obeys the contract with variable.
// The clone should attach to new_grad's history if GradMode::is_enabled().
inline at::Tensor clone_obey_contract(const at::Tensor& new_grad, const at::Tensor& variable) {
  if (variable.is_non_overlapping_and_dense()) {
    // (1)
    // Does this dicey-looking sequence attach the result to new_grad's
    // history if GradMode::is_enabled()?  Yes, and @alband says it should.
    return std::move(at::empty_strided(variable.sizes(), variable.strides(),
                                       variable.options().memory_format(c10::nullopt))
                     .copy_(new_grad));
  } else {
    // (2)
    return new_grad.clone(at::MemoryFormat::Contiguous);
  }
}

} // namespace utils
} // namespace autograd
} // namespace torch
