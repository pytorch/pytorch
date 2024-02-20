#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/ATen.h>

#include <string>

namespace torch::inductor {

struct TensorMetaInfo {
  bool is_symbolic;
  c10::ScalarType dtype;
  c10::Device device;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  TensorMetaInfo(
      bool is_symbolic,
      c10::ScalarType dtype,
      c10::Device device,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides);
  bool operator==(const TensorMetaInfo& other) const;
  static bool sanityCheck(const TensorMetaInfo& tensor_meta_info);
  static std::vector<TensorMetaInfo> fromConfig(
      const std::vector<std::string>&);
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
