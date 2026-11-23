#include <torch/csrc/stable/grad_mode.h>
#include <torch/csrc/stable/library.h>

#include <tuple>

std::tuple<bool, bool> my_auto_fw_grad_mode_guard(bool enabled) {
  torch::stable::AutoFwGradMode guard(enabled);
  return {torch_is_fw_grad_mode_enabled(), guard.prev_mode};
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_auto_fw_grad_mode_guard(bool enabled) -> (bool, bool)");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl(
      "my_auto_fw_grad_mode_guard", TORCH_BOX(&my_auto_fw_grad_mode_guard));
}
