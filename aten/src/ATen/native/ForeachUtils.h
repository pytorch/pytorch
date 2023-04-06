#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <ATen/Dispatch.h>
#include <ATen/ScalarType.h>
#include <ATen/Device.h>
#include <ATen/TypeDefault.h>
#include <c10/util/ArrayRef.h>
#include <ATen/native/utils/ParamsHash.h>
#include <unordered_map>
#include <utility>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/result_type_native.h>
#endif

namespace at {
namespace native {
namespace {
// Check if tensor list has either a boolean tensor or a integer tensor
bool has_integral_tensor(TensorList tensors, const bool includeBool) {
  return std::any_of(tensors.begin(), tensors.end(),
    [&includeBool](const auto & t) { return at::isIntegralType(t.scalar_type(), includeBool); });
}
// check if tensor list has bool tensors
bool has_bool_tensor(TensorList tensors) {
  return std::any_of(tensors.begin(), tensors.end(),
    [](const auto & t) -> bool { return t.scalar_type() == ScalarType::Bool; });
}

// Check foreach API restrictions
// - Tensor lists must be non-empty.
// - All TensorLists and ScalarLists must have the same number of elements.
// - Corresponding tensors must have the same size.
void check_foreach_api_restrictions(TensorList tensors) {
  TORCH_CHECK(!tensors.empty(), "Tensor list must have at least one tensor.");
}

void check_foreach_api_restrictions(TensorList tensors, ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors);
  TORCH_CHECK(tensors.size() == scalars.size(), "Tensor list must have same number of elements as scalar list.");
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(!tensors1.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(!tensors2.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors2.size());
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2, TensorList tensors3) {
  TORCH_CHECK(!tensors1.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(!tensors2.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(!tensors3.empty(), "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors2.size());
  TORCH_CHECK(tensors1.size() == tensors3.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors3.size());
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2, TensorList tensors3, ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  TORCH_CHECK(tensors1.size() == scalars.size(), "Tensor list must have same number of elements as scalar list, got ", tensors1.size(), " and ", scalars.size());
}

// To go via 'fast' path, several conditions must be satisfied
// - All tensors in all lists must have the same dtype.
// - All tensors must be on the same device
// - All tensors must have strided layout
// - All tensors must be non-overlapping and dense
// - Resulting tensor must have the same dtype as the input one

// Please, make sure to call check_foreach_api_restrictions before calling this method.
// There is a set of preconditions that have to be satisfied.
bool check_fast_path_restrictions(
  ArrayRef<TensorList> tensorLists,
  ArrayRef<Scalar> scalarList = {},
  bool does_op_promote_integer_inputs_to_float = false) {
    const auto expected_dtype = tensorLists[0][0].dtype();
    const auto expected_device = tensorLists[0][0].device();

    auto is_tensor_okay = [&](const Tensor& tensor) {
      return tensor.dtype() == expected_dtype &&
             tensor.device() == expected_device &&
             tensor.layout() == at::kStrided &&
             tensor.is_non_overlapping_and_dense();
    };

    for (const auto& tensorList : tensorLists) {
      for (const auto& tensor : tensorList) {
        if (!is_tensor_okay(tensor)) {
          return false;
        }
      }
    }

    // Check if corresponding tensors in tensor lists have the same sizes and strides.
    for (const auto& tensor_list : tensorLists) {
      for (const auto j : c10::irange(tensorLists[0].size())) {
        if (tensorLists[0][j].sizes() != tensor_list[j].sizes()) {
          return false;
        }
        if (tensorLists[0][j].strides() != tensor_list[j].strides()) {
          return false;
        }
      }
    }

    // This function has already checked that `tensorList[j][i]` for all j, i has the same dtype
    // using `is_tensor_okay` function above.
    // This means we only need to check if {tensorList[0][0], tensorList[0][1], tensorList[0][2], ...}
    // do type promotion with scalarLIst.
    for (const auto i : c10::irange(tensorLists[0].size())) {
      // For division, integer inputs will result in float.
      if (does_op_promote_integer_inputs_to_float) {
        if (at::isIntegralType(tensorLists[0][i].scalar_type(), /*includeBool*/ true)) {
          return false;
        }
      }
      if (!scalarList.empty()) {
        const auto& scalar = scalarList.size() == 1 ? scalarList[0] : scalarList[i];
        const auto& tensor = tensorLists[0][i];
        // note(mkozuki): This check might be responsible for `_foreach_add(bool_tensors, bool_tensors)`
        // being pushed to slow path.
        if (tensor.scalar_type() != at::native::result_type(scalar, tensor)) {
          return false;
        }
      }
    }

    return true;
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
          scalarList.emplace_back(c10::Scalar(scalar_data[i]));
        }
      });
  return scalarList;
}

bool can_use_fast_route(ArrayRef<TensorList> tensorLists,
                        ArrayRef<Scalar> scalarList = {},
                        bool does_op_promote_integer_inputs_to_float = false) {
  return check_fast_path_restrictions(tensorLists, scalarList, does_op_promote_integer_inputs_to_float);
}

bool can_use_fast_route(TensorList tensors1, TensorList tensors2, bool does_op_promote_integer_inputs_to_float = false) {
  return can_use_fast_route({tensors1, tensors2}, {}, does_op_promote_integer_inputs_to_float);
}

using nested_tensorvec_t = std::vector<std::vector<at::Tensor>>;
using scalartype_nested_tensorvec_map_t = std::unordered_map<at::ScalarType, nested_tensorvec_t>;
using device_scalartype_nested_tensorvec_map_t = std::unordered_map<at::Device, scalartype_nested_tensorvec_map_t>;
using nested_tensorvec_t = std::vector<std::vector<at::Tensor>>;
using scalartype_nested_tensorvec_map_t = std::unordered_map<at::ScalarType, nested_tensorvec_t>;
using device_scalartype_nested_tensorvec_map_t = std::unordered_map<at::Device, scalartype_nested_tensorvec_map_t>;
using DeviceDtypeKey = std::pair<at::Device, at::ScalarType>;
using IndicesT = std::vector<int>;
using nested_optional_tensorvec_t = std::vector<std::vector<c10::optional<at::Tensor>>>;
using TensorsAndIndicesT = std::pair<nested_optional_tensorvec_t, IndicesT>;
using FlatMap = std::unordered_map<DeviceDtypeKey, TensorsAndIndicesT, ParamsHash<DeviceDtypeKey>>;

FlatMap group_tensors_by_first_tensors_device_and_dtype(const nested_optional_tensorvec_t& nested_tensorlist, const bool with_indices) {
  FlatMap grouped_tensors_with_indices;

  TORCH_CHECK_GT(nested_tensorlist.size(), 0);
  TORCH_CHECK_GT(nested_tensorlist[0].size(), 0);
  const auto num_lists = nested_tensorlist.size();
  const auto num_tensors = nested_tensorlist[0].size();

  TORCH_CHECK(std::all_of(nested_tensorlist.cbegin(), nested_tensorlist.cend(),
    [&](const auto& tensorlist) -> bool {
      // note(crcrpar): Allow empty tensorlists following
      // ref: https://github.com/pytorch/pytorch/blob/85885301fd3c6adb8b9dc3cf7afadf6945566684/torch/utils/_foreach_utils.py#L21-L24
      return tensorlist.size() == num_tensors || tensorlist.size() == 0;
    }));

  for (const auto& tensor_index : c10::irange(num_tensors)) {
    const auto key = [&]() -> DeviceDtypeKey {
        const auto t = nested_tensorlist[0][tensor_index];
        TORCH_CHECK(t.has_value());
        return {t->device(), t->scalar_type()};
    }();
    if (!grouped_tensors_with_indices.count(key)) {
      grouped_tensors_with_indices.insert(
        {
          key,
          TensorsAndIndicesT{
            [&]() -> nested_optional_tensorvec_t {
              nested_optional_tensorvec_t nested_tensorvec;
              nested_tensorvec.reserve(num_lists);
             for (const auto& i : c10::irange(num_lists)) {
                std::vector<c10::optional<at::Tensor>> tensors;
                if (!nested_tensorlist[i].empty()) {
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
            }()
          }
        }
      );
    }
    for (const auto& list_index : c10::irange(num_lists)) {
      if (!nested_tensorlist[list_index].empty()) {
        grouped_tensors_with_indices[key].first[list_index].emplace_back(nested_tensorlist[list_index][tensor_index]);
      }
    }
    if (with_indices) {
      grouped_tensors_with_indices[key].second.emplace_back(tensor_index);
    }
  }

  return grouped_tensors_with_indices;
}

template<typename F>
device_scalartype_nested_tensorvec_map_t _group_tensors_by_device_and_scalartype(
    const std::vector<std::vector<at::Tensor>>& nested_tensorlists,
    F check_tensor
) {
  TORCH_CHECK_GT(nested_tensorlists.size(), 0);
  const auto num_lists{nested_tensorlists.size()};
  const auto num_tensors{nested_tensorlists[0].size()};
  for (const auto & i : c10::irange(num_lists)) {
    TORCH_CHECK_EQ(num_tensors, nested_tensorlists[i].size());
  }
  device_scalartype_nested_tensorvec_map_t grouped_tensors;
  for (const auto & tensor_index : c10::irange(num_tensors)) {
    const auto & first_tensor = nested_tensorlists[0][tensor_index];
    const auto device = first_tensor.device();
    const auto scalar_type = first_tensor.scalar_type();
    TORCH_CHECK(
        std::all_of(
          nested_tensorlists.cbegin(), nested_tensorlists.cend(),
          [&](const auto tensorlist) { return check_tensor(first_tensor, tensorlist[tensor_index]); }
          ));
    const auto gen_initializer = [&]() -> scalartype_nested_tensorvec_map_t::value_type {
      nested_tensorvec_t init_value;
      init_value.reserve(num_lists);
      for (const auto& tensorlist : nested_tensorlists) {
        init_value.emplace_back(std::vector<at::Tensor>{tensorlist[tensor_index]});
      }
      return {scalar_type, init_value};
    };
    if (!grouped_tensors.count(device)) {
      grouped_tensors[device] = {gen_initializer()};
    } else {
      if (!grouped_tensors[device].count(scalar_type)) {
        grouped_tensors[device].insert(gen_initializer());
      } else {
        for (const auto & i : c10::irange(num_lists)) {
          grouped_tensors[device][scalar_type][i].emplace_back(nested_tensorlists[i][tensor_index]);
        }
      }
    }
  }
  return grouped_tensors;
}


device_scalartype_nested_tensorvec_map_t group_tensors_by_device_and_scalartype(
    const std::vector<std::vector<Tensor>>& nested_tensorlists,
    const bool has_state_steps
) {
  return _group_tensors_by_device_and_scalartype(
      nested_tensorlists,
      [&](const at::Tensor& first_tensor, const at::Tensor& tensor) -> bool {
        return tensor.is_cuda() &&
          tensor.scalar_type() == first_tensor.scalar_type() &&
          tensor.device() == first_tensor.device() &&
          tensor.layout() == at::kStrided &&
          tensor.is_non_overlapping_and_dense() &&
          tensor.sizes() == first_tensor.sizes() &&
          tensor.strides() == first_tensor.strides();
      }
  );
}

} // namespace (anonymous)
}} // at::native
