#include <ATen/core/LazyTensor.h>

namespace at {

// We notify all siblings that they don't need to update this value
LazyTensorImpl::~LazyTensorImpl() {
  for (auto* sibling : siblings()) {
    if (!sibling) continue;
    for (size_t i = 0; i < sibling->siblings().size(); ++i) {
      auto rel = sibling->siblings()[i];
      if (rel == this) {
        sibling->siblings()[i] = nullptr;
      }
    }
  }
}

std::function<Tensor(LazyTensorImpl*)>& LazyTensorImpl::getResolver() {
  static std::function<Tensor(LazyTensorImpl*)> resolver_;
  return resolver_;
}

// Resolving a LazyTensor can be done in many ways,
// to_eager checks the registry for how.
// Note on mobile there is no TorchScript IR, so
// we register a different implementation.
Tensor LazyTensorImpl::to_eager() {
  return getResolver()(this);
}

} // namespace at
