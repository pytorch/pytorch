#include <ATen/OptionsGuard.h>
#include <ATen/optional.h>

namespace at {

thread_local at::optional<TensorOptions> DefaultTensorOptions::options_;

TensorOptions& DefaultTensorOptions::get() {
  if (!options_) {
    options_.emplace(
        /*use_thread_local_default_options=*/false);
  }
  return *options_;
}

} // namespace at
