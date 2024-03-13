#if !defined(C10_MOBILE) && !defined(ANDROID)

#include <torch/csrc/inductor/aoti_eager/kernel_checker.h>
#include <torch/csrc/inductor/aoti_eager/kernel_meta_info.h>

namespace torch::inductor {

TensorChecker::TensorChecker(const at::Tensor& src_tensor)
    : src_meta_info_(src_tensor) {}
TensorChecker::TensorChecker(const TensorMetaInfo& src_meta_info)
    : src_meta_info_(src_meta_info) {}

} // namespace torch::inductor
#endif
