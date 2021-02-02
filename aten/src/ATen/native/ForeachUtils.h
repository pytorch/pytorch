#pragma once
#include <ATen/ATen.h>

namespace at {
namespace native {
namespace {

// Check if tensor list has a boolean tensor
bool has_int_or_bool_tensor(TensorList tensors) {
    bool has_integral = false;
    for (auto t : tensors) {
        if (at::isIntegralType(t.scalar_type(), /*includeBool=*/true)) {
            has_integral = true;
        }
    }
    return has_integral;
}

void check_foreach_api_restrictions(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
}

void check_foreach_api_restrictions(TensorList tensors, ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors);
  TORCH_CHECK(tensors.size() == scalars.size(), "Tensor list must have same number of elements as scalar list.");
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors2.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors2.size());

  for (int i = 0; i < tensors1.size(); i++) {
    TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors2[i].sizes());
  }
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2, TensorList tensors3) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors2.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors3.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors2.size());
  TORCH_CHECK(tensors1.size() == tensors3.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors3.size());

  for (int i = 0; i < tensors1.size(); i++) {
    TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors2[i].sizes());
    TORCH_CHECK(tensors1[i].sizes() == tensors3[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors3[i].sizes());
  }
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2, TensorList tensors3, ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  TORCH_CHECK(tensors1.size() == scalars.size(), "Tensor list must have same number of elements as scalar list, got ", tensors1.size(), " and ", scalars.size());
}

// To go via 'fast' path, several conditions must be satisfied
// - All tensors must be on the same device
// - All tensors must have strided layout
// - All tensors must be non-overlapping and dense
// - Resulting tensor must have the same dtype as the input one

bool will_promote_tensor(const Tensor& tensor, Scalar scalar, bool does_op_promote_integer_inputs_to_float = false) {
  // complex scalar + integral or boolean tensor will result in complex tensor
  if (scalar.isComplex() && at::isIntegralType(tensor.scalar_type(), /*includeBool*/ true)) {
    return true;
  }

  // complex scalar + float tensor will result in complex tensor
  if (scalar.isComplex() && at::isFloatingType(tensor.scalar_type())) {
    return true;
  }

  // float scalar + integral or boolean tensor will result in float tensor
  if (scalar.isFloatingPoint() && at::isIntegralType(tensor.scalar_type(), /*includeBool*/ true)) {
    return true;
  }

  // integral scalar + boolean tensor will result in integral tensor
  if (scalar.isIntegral(/*includeBool*/ false) && tensor.dtype() == at::kBool) {
    return true;
  }

  // In case of division, integer inputs will result in float
  if (does_op_promote_integer_inputs_to_float) {
    if (at::isIntegralType(tensor.scalar_type(), /*includeBool*/ true)) {
      return true;
    }
  }

  return false;
}

// Please, make sure to call check_foreach_api_restrictions before calling this method. 
// There is a set of preconditions that have to be satisfied. 
bool check_fast_path_restrictions(
  ArrayRef<TensorList> tensorLists, 
  ArrayRef<Scalar> scalarList = {}, 
  bool does_op_promote_integer_inputs_to_float = false) {
    auto expected_device = tensorLists[0][0].device();
    auto expected_strides = tensorLists[0][0].strides();
    auto expected_dtype = tensorLists[0][0].dtype();

    auto is_tensor_okay = [&](const Tensor& tensor) {
      return tensor.dtype() == expected_dtype &&
             tensor.device() == expected_device &&
             tensor.layout() == at::kStrided &&
             tensor.strides() == expected_strides &&
             tensor.is_non_overlapping_and_dense();
    };

    for (const auto& tensorList : tensorLists) {
      for (const auto& tensor : tensorList) {
        if (!is_tensor_okay(tensor)) {
          return false;
        }
      }
    }

    // For all j, tensorList[j][0] have the same shape and dtype. (this was a precondition
    // checked by `check_foreach_api_restrictions`). This means we only need to check if
    // {tensorList[0][0], tensorList[0][1], tensorList[0][2], ...} do type promotion with scalarLIst.
    for (int i=0; i < tensorLists[0].size(); i++) {
      if (does_op_promote_integer_inputs_to_float) {
        if (at::isIntegralType(tensorLists[0][i].scalar_type(), /*includeBool*/ true)) {
          return false;
        }
      }

      if (scalarList.size() == 1) {
        if (will_promote_tensor(tensorLists[0][i], scalarList[0])) {
          return false;
        }
      } else if (scalarList.size() > 1) {
        // Complex scalar list is not supported due to the limit for kernel launch argument (4KB)
        if (scalarList[i].isComplex()) {
          return false;
        }

        if (will_promote_tensor(tensorLists[0][i], scalarList[i])) {
          return false;
        }
      }
    }

    return true;
}

bool can_use_fast_route(ArrayRef<TensorList> tensorLists, 
                        ArrayRef<Scalar> scalarList = {}, 
                        bool does_op_promote_integer_inputs_to_float = false) {
#ifdef __HIP_PLATFORM_HCC__
  return false;
#else
  return check_fast_path_restrictions(tensorLists, scalarList, does_op_promote_integer_inputs_to_float);
#endif
}

bool can_use_fast_route(TensorList tensors1, TensorList tensors2, bool does_op_promote_integer_inputs_to_float = false) {
#ifdef __HIP_PLATFORM_HCC__
  return false;
#else
  return can_use_fast_route({tensors1, tensors2}, {}, does_op_promote_integer_inputs_to_float);
#endif
}

}
}} // at::native
