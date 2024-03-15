#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <torch/csrc/inductor/aoti_eager/kernel_checker.h>
#include <torch/csrc/inductor/aoti_eager/kernel_meta_info.h>

namespace torch::inductor {

class StaticTensorChecker : public TensorChecker {
 public:
  StaticTensorChecker(const at::Tensor& tensor) : TensorChecker(tensor) {}
  StaticTensorChecker(const TensorMetaInfo& meta_info)
      : TensorChecker(meta_info) {}
  bool check(const at::Tensor& tensor) override;
  bool check(const TensorMetaInfo& meta_info) override;
};

} // namespace torch::inductor
#endif
