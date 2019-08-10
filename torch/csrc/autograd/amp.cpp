#include <torch/csrc/autograd/amp.h>

// Maybe these should go in ATen/native/amp.cpp?
namespace torch {
namespace autograd {
namespace amp {
namespace {
  bool enabled = false;
  float loss_scale = 1.0;
}

bool Amp::is_grad_scaling_enabled() {
  return enabled;
}

void Amp::set_grad_scaling_enabled(bool new_enabled) {
  enabled = new_enabled;
}

float Amp::get_grad_scale() {
  return loss_scale;
}

void Amp::set_grad_scale(float new_scale) {
  loss_scale = new_scale;
}
} // namespace torch
} // namespace autograd
} // namespace amp
