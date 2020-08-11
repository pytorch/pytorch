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

bool check_fast_route(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (auto t : tensors) {
    if (t.layout() != at::kStrided) {
      return false;
    }

    if (!t.is_non_overlapping_and_dense()) {
      return false;
    }

    if ((at::isIntegralType(t.scalar_type(), true) && scalar.isFloatingPoint()) || 
        t.scalar_type() == at::kBool) {
     return false;
    }
  }

  return true;
}
}} // at::native
