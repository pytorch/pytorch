#include <ATen/core/OptionsGuard.h>
#include <ATen/core/optional.h>

namespace at {

#if !AT_MOBILE

thread_local at::optional<TensorOptions> DefaultTensorOptions::options_;

TensorOptions& DefaultTensorOptions::get() {
  if (!options_) {
    options_.emplace(
        /*use_thread_local_default_options=*/false);
  }
  return *options_;
}

#else

TensorOptions DefaultTensorOptions::options_(/*use_thread_local_default_options=*/false);

const TensorOptions& DefaultTensorOptions::get() {
  return *options_;
}

#endif

} // namespace at
