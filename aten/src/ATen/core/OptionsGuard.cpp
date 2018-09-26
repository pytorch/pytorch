#include <ATen/core/OptionsGuard.h>
#include <ATen/core/optional.h>
#include <ATen/core/Layout.h>

namespace at {

#if !AT_MOBILE && !defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

thread_local at::optional<DefaultTensorOptions> DefaultTensorOptions::options_;

DefaultTensorOptions& DefaultTensorOptions::get() {
  if (!options_) {
    options_.emplace();
  }
  return *options_;
}

#else

DefaultTensorOptions DefaultTensorOptions::options_;

const DefaultTensorOptions& DefaultTensorOptions::get() {
  return options_;
}

#endif

} // namespace at
