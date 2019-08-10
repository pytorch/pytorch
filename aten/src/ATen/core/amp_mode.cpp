#include <ATen/core/amp_mode.h>

// Maybe these should go in ATen/native/amp.cpp?
namespace at {

// should these be thread_local?
namespace {
  bool enabled = false;
  float loss_scale = 1.0;
}

bool AmpMode::is_grad_scaling_enabled() {
  return enabled;
}

void AmpMode::set_grad_scaling_enabled(bool new_enabled) {
  enabled = new_enabled;
}

float AmpMode::get_grad_scale() {
  return loss_scale;
}

void AmpMode::set_grad_scale(float new_scale) {
  loss_scale = new_scale;
}
} // namespace torch
