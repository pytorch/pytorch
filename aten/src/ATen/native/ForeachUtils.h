#pragma once
#include <ATen/ATen.h>

namespace at {
namespace native {
namespace {

void verify_list(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  auto expected_dtype = tensors[0].dtype();

  for (auto t : tensors) {
    TORCH_CHECK(t.dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
  }
}

// To go via 'fast' path, several conditions must be satisfied 
// - All tensors must have strided layout
// - All tensors must be non-overlapping and dense
// - Resulting tensor must have the same dtype as the input one
bool check_fast_route(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  auto expected_device = tensors[0].device();

  for (auto t : tensors) {
    if (t.device() != expected_device) {
      return false;
    }

    if (t.layout() != at::kStrided) {
      return false;
    }

    if (!t.is_non_overlapping_and_dense()) {
      return false;
    }

    // complex scalar + integral or boolean tensor will result in complex tensor
    if (scalar.isComplex() && at::isIntegralType(t.scalar_type(), /*includeBool*/ true)) {
      return false;
    }

    // float scalar + integral or boolean tensor will result in float tensor
    if (scalar.isFloatingPoint() && at::isIntegralType(t.scalar_type(), /*includeBool*/ true)) {
      return false;
    }

    // integral scalar + boolean tensor will result in integral tensor 
    if (scalar.isIntegral(/*includeBool*/ false) && t.dtype() == at::kBool) {
      return false;
    }
  }

  return true;
}

} // namespace
}} // at::native
