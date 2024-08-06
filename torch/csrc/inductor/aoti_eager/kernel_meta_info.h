#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/ATen.h>
#include <c10/core/SymIntArrayRef.h>
#include <torch/csrc/dynamo/guards.h>

#include <string>

namespace torch::inductor {

// Regarding a aten operation implemented by AOTI, the metadata of the input
// tensors will be cached on the disk to acclerate next run. TensorMetada
// structure is to represent the metadata of each input tensor. it includes
// whether the tensor is symbolic, the dtype, the device, the sizes and the
// strides of the tensor. When the metadata of the input tensors is the same as
// the cached metadata, the cached kernel library will be loaded and executed.
// Otherwise, the AOT Inductor will be called again to generate the kernel
// library.
// Beyond the TensorMetadata, we build guard/TensorCheck for each input tensor
// as well to support symbolic shape. We intend to utilize TensorCheck to find
// out the proper kernel rather than TensorMetada comparison. Suppose an
// operation with a single input tensor and two kernels:
//   kernel1: TensorMetadata(is_symbolic=false, dtype=Float, device=CPU,
//   sizes=[s0, s1, s2], strides=[s1 * s2, s2, 1]) kernel2:
//   TensorMetadata(is_symbolic=false, dtype=Float, device=CPU, sizes=[3, s1,
//   s2], strides=[s1 * s2, s2, 1])
// If a tensor with sizes=[3, 4, 5] is passed to the operation, both kernel1 and
// kernel2 support the tensor shape. In this case, we need to use TensorCheck
// plus some heruistic rules to find out the proper kernel.
struct TensorMetadata {
  // Indicate whether the tensor is symbolic and it may be concluded by sizes_
  // and strides_ in the future.
  bool is_symbolic_;
  // Dtype of a tensor(For scalar, we will wrap it as a scalar tensor)
  c10::ScalarType dtype_ = c10::ScalarType::Undefined;
  // Device of a tensor.
  c10::Device device_;
  // Dispatch key set of a tensor
  c10::DispatchKeySet dispatch_key_set_;
  // Sizes of a tensor. Currently, we only support static shape and use int64_t
  // to represent the sizes. In the future, we will create symbolic size and use
  // SymInt to represent it to support symbolic shape.
  std::vector<int64_t> sizes_;
  // Strides of a tensor. For symbolic shape support, it is the same as sizes_
  std::vector<int64_t> strides_;
  // requires grad
  bool requires_grad_ = false;
  // TensorCheck for the tensor
  std::optional<dynamo::TensorCheck> tensor_check_;

  TensorMetadata()
      : is_symbolic_(false),
        device_(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES),
        sizes_({}),
        strides_({}) {}
  TensorMetadata(const at::Tensor& src_tensor);
  TensorMetadata(
      bool is_symbolic,
      c10::ScalarType dtype,
      c10::Device device,
      c10::DispatchKeySet dispatch_key_set,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides,
      bool requires_grad = false);

  // Build TensorCheck for the tensor by using the data fields in TensorMetadata
  void build_guard(const dynamo::LocalState& local_state);

  // Compare two TensorMetadata objects
  bool operator==(const TensorMetadata& other) const;
};

// ParameterTag is to represent the type of the input parameters of a aten
// operation. Currently, we support the following types:
//   1. TENSOR: a single tensor
//   2. TENSOR_OPTIONAL: a single optional tensor
//   3. TENSOR_LIST: a list of tensors
//   4. TENSOR_LIST_OPTIONAL: a list of optional tensors
//   5. SCALAR: a scalar value
// If we need to support more types in the future, we will add more types in the
// ParameterTag enum. For example, we will extend the enum to support string,
// Dimname and so on to support more types of input parameters of aten
// operations.
enum ParameterTag {
  TENSOR,
  TENSOR_OPTIONAL,
  TENSOR_LIST,
  TENSOR_LIST_OPTIONAL,
  SCALAR,
  STRING,
  DEVICE,
  INVALID,
};

// ParameterMetadataValue is to represent the value of the input parameters of a
// aten operation.
using ParameterMetadataValue = std::variant<
    TensorMetadata,
    std::vector<TensorMetadata>,
    c10::Scalar,
    std::string,
    c10::Device>;

// ParameterMetadata is to represent the metadata of the input parameters of a
// aten operation. It includes the tag of the parameter, the value of the
// parameter and the order of the parameter.
struct ParameterMetadata {
  // The tag of the parameter. It indicates the type of the parameter.
  ParameterTag tag_;
  // The value of the parameter. It can be a tensor, a list of tensors or a
  // scalar.
  ParameterMetadataValue value_;
  // The order of the parameter is used to distinguish the parameters with the
  // same tag. For example, an operation with two input tensors, the first
  // tensor is a optional tensor and the second tensor is a tensor. The first
  // tensor will have the order 0 and the second tensor will have the order 1.
  uint64_t order_{};

  ParameterMetadata() : tag_(INVALID) {}
  ParameterMetadata(TensorMetadata tensor_metadata, uint64_t input_order);
  ParameterMetadata(const at::Tensor& tensor, uint64_t input_order);
  ParameterMetadata(
      const std::vector<at::Tensor>& tensor_list,
      uint64_t input_order);
  ParameterMetadata(
      const std::vector<TensorMetadata>& tensor_metadata_list,
      uint64_t input_order);
  ParameterMetadata(const c10::Scalar& scalar, uint64_t input_order);
  ParameterMetadata(const std::string& string_value, uint64_t input_order);
  ParameterMetadata(const c10::Device& device, uint64_t input_order);

  bool operator==(const ParameterMetadata& other) const;

 private:
  // Helper function to compare two ParameterMetadata objects with the same
  // SCALAR tag.
  bool equal_to(const c10::Scalar& scalar) const;
};

} // namespace torch::inductor
#endif
