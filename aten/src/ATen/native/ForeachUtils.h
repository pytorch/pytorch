#pragma once
#include <ATen/ATen.h>

namespace at { 
namespace native {

void verify_list(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  auto expected_dtype = tensors[0].dtype();
  auto expected_device = tensors[0].device();
  auto expected_sizes = tensors[0].sizes();

  for (auto t : tensors) {
    TORCH_CHECK(t.sizes() == expected_sizes, "All tensors in the tensor list must have the same size.");
    TORCH_CHECK(t.dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(t.device() == expected_device, "All tensors in the tensor list must have the same device.");
  }
}

// In order to go via 'fast' path, sevelar conditions must be satisfied 
// - All tensors must have strided layout
// - All tensors must be non overlapping and dense
// - Resulting tensor must have the same dtype as the input one
bool check_fast_route(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (auto t : tensors) {
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
}} // at::native
