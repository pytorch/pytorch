#include <torch/csrc/stable/inference_mode.h>
#include <torch/csrc/stable/library.h>

bool my_inference_mode_guard(bool enabled) {
  torch::stable::InferenceMode guard(enabled);
  return torch::stable::InferenceMode::is_enabled();
}

bool my_default_inference_mode_guard() {
  torch::stable::InferenceMode guard;
  return torch::stable::InferenceMode::is_enabled();
}

bool my_is_inference_mode_enabled() {
  return torch::stable::InferenceMode::is_enabled();
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_inference_mode_guard(bool enabled) -> bool");
  m.def("my_default_inference_mode_guard() -> bool");
  m.def("my_is_inference_mode_enabled() -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_inference_mode_guard", TORCH_BOX(&my_inference_mode_guard));
  m.impl(
      "my_default_inference_mode_guard",
      TORCH_BOX(&my_default_inference_mode_guard));
  m.impl(
      "my_is_inference_mode_enabled",
      TORCH_BOX(&my_is_inference_mode_enabled));
}
