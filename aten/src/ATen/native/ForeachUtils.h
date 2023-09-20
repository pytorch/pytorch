#pragma once

#include <ATen/Device.h>
#include <ATen/Dispatch.h>
#include <ATen/ScalarType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/utils/ParamsHash.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/result_type_native.h>
#endif

#include <unordered_map>
#include <vector>

namespace at::native {
namespace {
// Check if tensor list has either a boolean tensor or a integer tensor
bool has_integral_tensor(TensorList tensors, const bool includeBool) {
  return std::any_of(
      tensors.begin(), tensors.end(), [&includeBool](const auto& t) {
        return at::isIntegralType(t.scalar_type(), includeBool);
      });
}
// check if tensor list has bool tensors
bool has_bool_tensor(TensorList tensors) {
  return std::any_of(tensors.begin(), tensors.end(), [](const auto& t) -> bool {
    return t.scalar_type() == ScalarType::Bool;
  });
}

// Check foreach API restrictions
// - Tensor lists must be non-empty.
// - All TensorLists and ScalarLists must have the same number of elements.
// - Corresponding tensors must have the same size.
void check_foreach_api_restrictions(TensorList tensors) {
  TORCH_CHECK(!tensors.empty(), "Tensor list must have at least one tensor.");
}

void check_foreach_api_restrictions(
    TensorList tensors,
    ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors);
  TORCH_CHECK(
      tensors.size() == scalars.size(),
      "Tensor list must have same number of elements as scalar list.");
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(!tensors1.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(!tensors2.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(
      tensors1.size() == tensors2.size(),
      "Tensor lists must have the same number of tensors, got ",
      tensors1.size(),
      " and ",
      tensors2.size());
}

void check_foreach_api_restrictions(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  TORCH_CHECK(!tensors1.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(!tensors2.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(!tensors3.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(
      tensors1.size() == tensors2.size(),
      "Tensor lists must have the same number of tensors, got ",
      tensors1.size(),
      " and ",
      tensors2.size());
  TORCH_CHECK(
      tensors1.size() == tensors3.size(),
      "Tensor lists must have the same number of tensors, got ",
      tensors1.size(),
      " and ",
      tensors3.size());
}

void check_foreach_api_restrictions(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3,
    ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  TORCH_CHECK(
      tensors1.size() == scalars.size(),
      "Tensor list must have same number of elements as scalar list, got ",
      tensors1.size(),
      " and ",
      scalars.size());
}

// Helper function called in check_fast_path_restrictions to check whether all
// corresponding tensors (aligning in index across the tensorLists) share the
// same device and dtype.
bool _check_tensors_share_device_and_dtype(ArrayRef<TensorList> tensorLists) {
  const auto expected_dtype = tensorLists[0][0].dtype();
  const auto expected_device = tensorLists[0][0].device();

  auto is_tensor_okay = [&](const Tensor& tensor) {
    return tensor.dtype() == expected_dtype &&
        tensor.device() == expected_device && tensor.layout() == at::kStrided &&
        tensor.is_non_overlapping_and_dense();
  };

  for (const auto& tensorList : tensorLists) {
    for (const auto& tensor : tensorList) {
      if (!is_tensor_okay(tensor)) {
        return false;
      }
    }
  }

  return true;
}

// Helper function called in check_fast_path_restrictions to check if
// corresponding tensors in tensor lists have the same sizes and strides.
bool _check_tensors_share_sizes_and_strides(ArrayRef<TensorList> tensorLists) {
  for (const auto i : c10::irange(1, tensorLists.size())) {
    for (const auto j : c10::irange(tensorLists[0].size())) {
      if (tensorLists[0][j].sizes() != tensorLists[i][j].sizes() ||
          tensorLists[0][j].strides() != tensorLists[i][j].strides()) {
        return false;
      }
    }
  }

  return true;
}

// Helper function called in check_fast_path_restrictions to check whether
// all tensors type promote properly with the scalars in scalarList. This
// function assumes that _check_tensors_share_device_and_dtype has already been
// called so that all corresponding tensors in tensorLists have the same dtype.
// Then, it is sufficient to check the type promotion with just one tensorList.
bool _check_tensors_do_type_promotion_with_scalars(
    TensorList tensorList,
    ArrayRef<Scalar> scalarList = {},
    bool does_op_promote_integer_inputs_to_float = false) {
  for (const auto i : c10::irange(tensorList.size())) {
    // For division, integer inputs will result in float.
    if (does_op_promote_integer_inputs_to_float) {
      if (at::isIntegralType(
              tensorList[i].scalar_type(), /*includeBool*/ true)) {
        return false;
      }
    }
    if (!scalarList.empty()) {
      const auto& scalar =
          scalarList.size() == 1 ? scalarList[0] : scalarList[i];
      const auto& tensor = tensorList[i];
      // note(mkozuki): This check might be responsible for
      // `_foreach_add(bool_tensors, bool_tensors)` being pushed to slow path.
      if (tensor.scalar_type() != at::native::result_type(scalar, tensor)) {
        return false;
      }
    }
  }

  return true;
}

// To go via 'fast' path, several conditions must be satisfied
// - All tensors in all lists must have the same dtype.
// - All tensors must be on the same device
// - All tensors must have strided layout
// - All tensors must be non-overlapping and dense
// - Resulting tensor must have the same dtype as the input one

// Please, make sure to call check_foreach_api_restrictions before calling this
// method. There is a set of preconditions that have to be satisfied.
bool check_fast_path_restrictions(
    ArrayRef<TensorList> tensorLists,
    ArrayRef<Scalar> scalarList = {},
    bool does_op_promote_integer_inputs_to_float = false) {
  return _check_tensors_share_device_and_dtype(tensorLists) &&
      _check_tensors_share_sizes_and_strides(tensorLists) &&
      _check_tensors_do_type_promotion_with_scalars(
             tensorLists[0],
             scalarList,
             does_op_promote_integer_inputs_to_float);
}

std::vector<c10::Scalar> convert_tensor_to_scalar_list(
    const Tensor& scalarList_,
    int64_t expect_length) {
  std::vector<c10::Scalar> scalarList;
  TORCH_CHECK(
      scalarList_.device() == c10::kCPU,
      "Expected scalars to be on CPU, got ",
      scalarList_.device(),
      " instead.");
  TORCH_CHECK(
      scalarList_.is_contiguous(), "Expected scalars to be contiguous.");
  TORCH_CHECK(
      scalarList_.dim() == 1,
      "Expected packed scalar Tensor to be of dimension 1. Got ",
      scalarList_.dim(),
      " instead.");
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16,
      scalarList_.scalar_type(),
      "convert_tensor_to_scalar_list",
      [&]() {
        const scalar_t* scalar_data = scalarList_.data_ptr<scalar_t>();
        TORCH_CHECK(
            (expect_length == scalarList_.size(0)),
            "Expected length of scalars to match input of length ",
            expect_length,
            " but got ",
            scalarList_.size(0),
            " instead.");
        for (int64_t i = 0; i < scalarList_.size(0); i++) {
          scalarList.push_back(c10::Scalar(scalar_data[i]));
        }
      });
  return scalarList;
}

bool can_use_fast_route(
    ArrayRef<TensorList> tensorLists,
    ArrayRef<Scalar> scalarList = {},
    bool does_op_promote_integer_inputs_to_float = false) {
  return check_fast_path_restrictions(
      tensorLists, scalarList, does_op_promote_integer_inputs_to_float);
}

bool can_use_fast_route(
    TensorList tensors1,
    TensorList tensors2,
    bool does_op_promote_integer_inputs_to_float = false) {
  return can_use_fast_route(
      {tensors1, tensors2}, {}, does_op_promote_integer_inputs_to_float);
}

using DeviceDtypeKey = std::pair<at::Device, at::ScalarType>;
using IndicesT = std::vector<int>;
using nested_optional_tensorvec_t =
    std::vector<std::vector<c10::optional<at::Tensor>>>;
using TensorsAndIndicesT = std::pair<nested_optional_tensorvec_t, IndicesT>;
using FlatMap = std::unordered_map<
    DeviceDtypeKey,
    TensorsAndIndicesT,
    ParamsHash<DeviceDtypeKey>>;

FlatMap _group_tensors_by_first_tensors_device_and_dtype(
    const nested_optional_tensorvec_t& nested_tensorlist,
    const bool with_indices) {
  FlatMap grouped_tensors_with_indices;

  TORCH_CHECK(nested_tensorlist.size() > 0);
  TORCH_CHECK(nested_tensorlist[0].size() > 0);
  const auto num_lists = nested_tensorlist.size();
  const auto num_tensors = nested_tensorlist[0].size();

  TORCH_CHECK(std::all_of(
      nested_tensorlist.cbegin(),
      nested_tensorlist.cend(),
      [&](const auto& tensorlist) -> bool {
        // note(crcrpar): Allow empty tensorlists following
        // ref:
        // https://github.com/pytorch/pytorch/blob/85885301fd3c6adb8b9dc3cf7afadf6945566684/torch/utils/_foreach_utils.py#L21-L24
        return tensorlist.size() == num_tensors || tensorlist.size() == 0;
      }));

  for (const auto& tensor_index : c10::irange(num_tensors)) {
    const auto key = [&]() -> DeviceDtypeKey {
      const auto t = nested_tensorlist[0][tensor_index];
      TORCH_CHECK(
          t.has_value(),
          "Tensors of the first list of nested Tensor lists are supposed to be defined but ",
          "the ",
          tensor_index,
          "-th Tensor is not.");
      return {t->device(), t->scalar_type()};
    }();
    TORCH_CHECK(
        std::all_of(
            nested_tensorlist.cbegin(),
            nested_tensorlist.cend(),
            [&](const auto& tensorlist) -> bool {
              if (tensorlist.size() == 0) {
                return true;
              }
              const auto& tensor = tensorlist[tensor_index];
              // note(crcrpar): Currently the scope of this function is
              // optimizers so there could be `state_steps` and other scalars
              // whose elements are float tensors no matter what the parameter's
              // dtype is.
              if (!tensor.has_value()) {
                return true;
              } else {
                const auto s = tensor->scalar_type();
                const auto d = tensor->device();
                // Note: `step` or `state_step` is float32 by default.
                if (key.first == d) {
                  return key.second == s || s == at::ScalarType::Float;
                } else if (d.is_cpu()) {
                  // note(crcrpar): There are some test cases (e.g.
                  // TestOptim::test_adam) where state_steps are on CPU and the
                  // others are on CUDA. Currently a state_step Tensor has the
                  // dtype of float.
                  return s == at::ScalarType::Float;
                } else {
                  return false;
                }
              }
            }),
        "Tensors of the same index must be on the same device and the same dtype except `step` tensors that can be CPU and float32 notwithstanding");
    if (!grouped_tensors_with_indices.count(key)) {
      grouped_tensors_with_indices.insert(
          {key,
           TensorsAndIndicesT{
               [&]() -> nested_optional_tensorvec_t {
                 nested_optional_tensorvec_t nested_tensorvec;
                 nested_tensorvec.reserve(num_lists);
                 for (const auto& i : c10::irange(num_lists)) {
                   std::vector<c10::optional<at::Tensor>> tensors;
                   if (!nested_tensorlist[i].empty()) {
                     // NB: num_tensors is the max possible length for any of
                     // the inner lists of tensor references. Reserving the max
                     // trades memory for perf. This should not have significant
                     // impact.
                     tensors.reserve(num_tensors);
                   }
                   nested_tensorvec.emplace_back(tensors);
                 }
                 return nested_tensorvec;
               }(),
               [&]() -> IndicesT {
                 if (!with_indices) {
                   return {};
                 } else {
                   IndicesT indices;
                   indices.reserve(num_tensors);
                   return indices;
                 }
               }()}});
    }
    for (const auto& list_index : c10::irange(num_lists)) {
      if (!nested_tensorlist[list_index].empty()) {
        grouped_tensors_with_indices[key].first[list_index].emplace_back(
            nested_tensorlist[list_index][tensor_index]);
      }
    }
    if (with_indices) {
      grouped_tensors_with_indices[key].second.emplace_back(tensor_index);
    }
  }

  return grouped_tensors_with_indices;
}

} // namespace
} // namespace at::native
