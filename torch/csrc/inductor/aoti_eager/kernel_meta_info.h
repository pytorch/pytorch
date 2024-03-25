#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/ATen.h>
#include <c10/core/SymIntArrayRef.h>

#include <nlohmann/json.hpp>
#include <string>

namespace torch::inductor {

class TensorChecker;

struct TensorMetaInfo {
  bool is_symbolic;
  c10::ScalarType dtype;
  c10::Device device;
  std::vector<c10::SymInt> sizes;
  std::vector<c10::SymInt> strides;
  std::shared_ptr<TensorChecker> tensor_checker;

  TensorMetaInfo(const at::Tensor& src_tensor);
  TensorMetaInfo(
      bool is_symbolic,
      c10::ScalarType dtype,
      c10::Device device,
      std::vector<c10::SymInt> sizes,
      std::vector<c10::SymInt> strides);

  bool operator==(const TensorMetaInfo& other) const;
  static bool sanityCheck(const TensorMetaInfo& tensor_meta_info);
  static std::vector<TensorMetaInfo> loadFromFile(
      const std::vector<nlohmann::json>&);
};

struct TensorMetaInfoHash {
  size_t operator()(const TensorMetaInfo&) const;
};

using AOTIKernelMetaInfo = std::vector<TensorMetaInfo>;

struct AOTIKernelMetaInfoHash {
  size_t operator()(const AOTIKernelMetaInfo&) const;
};

} // namespace torch::inductor
#endif
