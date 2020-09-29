#pragma once
#include <ATen/ATen.h>

namespace at {
namespace native {
namespace {

// Set of foreach API restrictions
// - All tensors must be of the same dtype
// - All corresponding tensors must be of the same size
void check_foreach_api_restrictions(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  auto expected_dtype = tensors[0].dtype();

  for (const auto& t : tensors) {
    TORCH_CHECK(t.dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
  }
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors2.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors2.size());

  auto expected_dtype = tensors1[0].dtype();

  for (int i = 0; i < tensors1.size(); i++) {
    TORCH_CHECK(tensors1[i].dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(tensors2[i].dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors2[i].sizes());
  }
}

void check_foreach_api_restrictions(TensorList tensors, ArrayRef<double> scalars) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(scalars.size() > 0, "Scalars list must have at least one value.");
  TORCH_CHECK(tensors.size() == scalars.size(), "Tensor list must have same number of elements as scalar list.");
}

// To go via 'fast' path, several conditions must be satisfied
// - All tensors must be on the same device
// - All tensors must have strided layout
// - All tensors must be non-overlapping and dense
// - All tensors must be on the same device
// - Resulting tensor must have the same dtype as the input one
bool can_use_fast_route(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  auto expected_device = tensors[0].device();

  for (auto t : tensors) {
    if (t.device() != expected_device) {
      return false;
    }

    if (t.layout() != at::kStrided) {
      return false;
    }

    if (t.device() != expected_device) {
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

bool can_use_fast_route(TensorList tensors1, TensorList tensors2) {
  auto expected_device = tensors1[0].device();

  for (int64_t i = 0; i < tensors1.size(); i++) {
    TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors from tensor lists have different size.");

    if (tensors1[i].device() != expected_device || 
        tensors2[i].device() != expected_device) {
      return false;
    }

    if (tensors1[i].layout() != at::kStrided || 
        tensors2[i].layout() != at::kStrided) {
      return false;
    }

    if (tensors1[i].device() != expected_device || 
        tensors2[i].device() != expected_device) {
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

bool can_use_fast_route(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  auto expected_device = tensors[0].device();

   for (auto t : tensors) {
    if (t.layout() != at::kStrided) {
      return false;
    }

    if (!t.is_non_overlapping_and_dense()) {
      return false;
    }

    if (t.device() != expected_device) {
      return false;
    }
  }

  return true;
}

bool can_use_fast_route(TensorList tensors, ArrayRef<double> scalars) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(scalars.size() > 0, "Scalars list must have at least one value.");
  TORCH_CHECK(tensors.size() == scalars.size(), "Tensor list must have same number of elements as scalar list.");

  return can_use_fast_route(tensors);
}

}
}} // at::native
