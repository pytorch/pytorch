#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_eager/static_checker.h>

namespace torch::inductor {

bool StaticTensorChecker::check(const at::Tensor& tensor) {
  return check(TensorMetaInfo(tensor));
}

bool StaticTensorChecker::check(const TensorMetaInfo& meta_info) {
  if (src_meta_info_.device.type() != meta_info.device.type()) {
    // TODO: Should we check device index here?
    return false;
  }

  if (src_meta_info_.dtype != meta_info.dtype) {
    return false;
  }

  if (src_meta_info_.sizes != meta_info.sizes) {
    return false;
  }

  if (src_meta_info_.strides != meta_info.strides) {
    return false;
  }

  return true;
}

} // namespace torch::inductor
#endif
