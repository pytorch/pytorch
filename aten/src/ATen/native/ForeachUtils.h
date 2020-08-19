#pragma once
#include <ATen/ATen.h>

namespace at { 
namespace native {
namespace {

void verify_list(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  auto expected_dtype = tensors[0].dtype();
  auto expected_device = tensors[0].device();

  for (const auto& t : tensors) {
    TORCH_CHECK(t.dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(t.device() == expected_device, "All tensors in the tensor list must have the same device.");
  }
}

// To go via 'fast' path, several conditions must be satisfied
// - All tensors must have strided layout
// - All tensors must be non-overlapping and dense
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

void verify_list(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors2.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors.");

  auto expected_dtype = tensors1[0].dtype();
  auto expected_device = tensors1[0].device();

  for (int i = 0; i < tensors1.size(); i++) {
    TORCH_CHECK(tensors1[i].dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(tensors1[i].device() == expected_device, "All tensors in the tensor list must have the same device.");
    
    TORCH_CHECK(tensors2[i].dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(tensors2[i].device() == expected_device, "All tensors in the tensor list must have the same device.");

    TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors in lists must have the same size.");
  }
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
