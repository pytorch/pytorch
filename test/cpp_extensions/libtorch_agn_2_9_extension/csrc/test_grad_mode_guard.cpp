#include <torch/csrc/stable/grad_mode.h>
#include <torch/csrc/stable/library.h>

#include <tuple>

void my_grad_mode_set_enabled(bool enabled) {
  torch::stable::GradMode::set_enabled(enabled);
}

bool my_grad_mode_is_enabled() {
  return torch::stable::GradMode::is_enabled();
}

std::tuple<bool, bool> my_auto_grad_mode_guard(bool enabled) {
  torch::stable::AutoGradMode guard(enabled);
  return {torch::stable::GradMode::is_enabled(), guard.prev_mode};
}

std::tuple<bool, bool> my_no_grad_guard() {
  torch::stable::NoGradGuard guard;
  return {torch::stable::GradMode::is_enabled(), guard.prev_mode};
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_grad_mode_set_enabled(bool enabled) -> ()");
  m.def("my_grad_mode_is_enabled() -> bool");
  m.def("my_auto_grad_mode_guard(bool enabled) -> (bool, bool)");
  m.def("my_no_grad_guard() -> (bool, bool)");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_grad_mode_set_enabled", TORCH_BOX(&my_grad_mode_set_enabled));
  m.impl("my_grad_mode_is_enabled", TORCH_BOX(&my_grad_mode_is_enabled));
  m.impl("my_auto_grad_mode_guard", TORCH_BOX(&my_auto_grad_mode_guard));
  m.impl("my_no_grad_guard", TORCH_BOX(&my_no_grad_guard));
}
