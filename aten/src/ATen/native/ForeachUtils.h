#pragma once
#include <ATen/ATen.h>

namespace at { 
namespace native {
namespace {

void verify_list(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  auto expected_dtype = tensors[0].dtype();
  auto expected_device = tensors[0].device();
  auto expected_sizes = tensors[0].sizes();

  for (const auto& t : tensors) {
    TORCH_CHECK(t.sizes() == expected_sizes, "All tensors in the tensor list must have the same size.");
    TORCH_CHECK(t.dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(t.device() == expected_device, "All tensors in the tensor list must have the same device.");
  }
}

void verify_list(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors2.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors.");

  auto expected_dtype = tensors1[0].dtype();
  auto expected_device = tensors1[0].device();
  auto expected_sizes = tensors1[0].sizes();

  for (int i = 0; i < tensors1.size(); i++) {
    TORCH_CHECK(tensors1[i].sizes() == expected_sizes, "All tensors in the tensor list must have the same size.");
    TORCH_CHECK(tensors1[i].dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(tensors1[i].device() == expected_device, "All tensors in the tensor list must have the same device.");
    TORCH_CHECK(tensors2[i].sizes() == expected_sizes, "All tensors in the tensor list must have the same size.");
    TORCH_CHECK(tensors2[i].dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(tensors2[i].device() == expected_device, "All tensors in the tensor list must have the same device.");
  }
}

bool check_fast_route(TensorList tensors, Scalar scalar) {
  for (const auto& t : tensors) {
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

bool check_fast_route(TensorList tensors1, TensorList tensors2) {
  for (int64_t i = 0; i < tensors1.size(); i++) {
    TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors from tensor lists have different size.");

    if (tensors1[i].layout() != at::kStrided || 
        tensors2[i].layout() != at::kStrided) {
      return false;
    }

    if (tensors1[i].strides() != tensors2[i].strides()) {
      return false;
    }

    if (!tensors1[i].is_non_overlapping_and_dense() || 
        !tensors2[i].is_non_overlapping_and_dense()) {
      return false;
    }
  }

  return true;
}

}
}} // at::native
