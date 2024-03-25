#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/ATen.h>

#include <torch/csrc/inductor/aoti_eager/kernel_meta_info.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/utils/python_dispatch.h>

#include <string>

namespace torch::inductor {

class TensorChecker {
 public:
  TensorChecker(const at::Tensor& src_tensor);
  TensorChecker(const TensorMetaInfo& src_meta_info);
  virtual ~TensorChecker() = default;
  virtual bool check(const at::Tensor& dst_tensor) = 0;
  virtual bool check(const TensorMetaInfo& dst_meta_info) = 0;

 protected:
  TensorMetaInfo src_meta_info_;
};

} // namespace torch::inductor
#endif
