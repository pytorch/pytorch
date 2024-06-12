#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/ATen.h>
#include <c10/core/SymIntArrayRef.h>

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
  c10::ScalarType dtype_;
  // Concrete scalar value. Serve for operations w/ scalar parameter
  c10::IValue scalar_value_;
  // Device of a tensor.
  c10::Device device_;
  // Sizes of a tensor. Currently, we only support static shape and use int64_t
  // to represent the sizes. In the future, we will create symbolic size and use
  // SymInt to represent it to support symbolic shape.
  std::vector<int64_t> sizes_;
  // Strides of a tensor. For symbolic shape support, it is the same as sizes_
  std::vector<int64_t> strides_;

  TensorMetadata(const at::Tensor& src_tensor);
  TensorMetadata(
      bool is_symbolic,
      c10::ScalarType dtype,
      c10::Device device,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides);
  TensorMetadata(
      bool is_symbolic,
      c10::ScalarType dtype,
      c10::IValue scalar_value,
      c10::Device device,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides);

  bool operator==(const TensorMetadata& other) const;
};

struct TensorMetadataHash {
  size_t operator()(const TensorMetadata&) const;
};

using AOTIKernelMetadata = std::vector<TensorMetadata>;

struct AOTIKernelMetadataHash {
  size_t operator()(const AOTIKernelMetadata&) const;
};

} // namespace torch::inductor
#endif
