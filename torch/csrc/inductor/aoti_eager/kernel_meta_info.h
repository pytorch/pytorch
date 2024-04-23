#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/ATen.h>
#include <c10/core/SymIntArrayRef.h>

#include <string>

namespace torch::inductor {

class TensorChecker;

struct TensorMetadata {
  bool is_symbolic_;
  c10::ScalarType dtype_;
  c10::IValue scalar_value_;
  c10::Device device_;
  std::vector<c10::SymInt> sizes_;
  std::vector<c10::SymInt> strides_;

  TensorMetadata(const at::Tensor& src_tensor);
  TensorMetadata(
      bool is_symbolic,
      c10::ScalarType dtype,
      c10::Device device,
      std::vector<c10::SymInt> sizes,
      std::vector<c10::SymInt> strides);
  TensorMetadata(
      bool is_symbolic,
      c10::ScalarType dtype,
      c10::IValue scalar_value,
      c10::Device device,
      std::vector<c10::SymInt> sizes,
      std::vector<c10::SymInt> strides);

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
