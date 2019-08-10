#include <torch/csrc/autograd/amp.h>

namespace torch {
namespace autograd {
namespace amp {
namespace {
  bool enabled = false;
  float loss_scale = 1.0;
}

bool getGradScalingEnabled() {
  return enabled;
}

void setGradScalingEnabled(bool new_enabled) {
  enabled = new_enabled;
}

float getGradScale() {
  return loss_scale;
}

void setGradScale(float new_scale) {
  loss_scale = new_scale;
}
} // namespace torch
} // namespace autograd
} // namespace amp
