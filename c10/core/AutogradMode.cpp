#include <c10/core/AutogradMode.h>

namespace c10 {

// By default, both grad modes are enabled and inference mode is disabled
thread_local uint8_t autograd_mode_bitfield = AutogradMode::GRAD_MODE_MASK | AutogradMode::FW_GRAD_MODE_MASK;

uint8_t AutogradMode::get_mode() {
  return autograd_mode_bitfield;
}

void AutogradMode::set_mode(uint8_t flag) {
  autograd_mode_bitfield = flag;
}

void AutogradMode::set_grad_mode(bool enabled) {
  if (enabled) {
    autograd_mode_bitfield |= AutogradMode::GRAD_MODE_MASK;
  } else {
    autograd_mode_bitfield &= ~AutogradMode::GRAD_MODE_MASK;
  }
}

void AutogradMode::set_fw_grad_mode(bool enabled) {
  if (enabled) {
    autograd_mode_bitfield |= AutogradMode::FW_GRAD_MODE_MASK;
  } else {
    autograd_mode_bitfield &= ~AutogradMode::FW_GRAD_MODE_MASK;
  }
}

void AutogradMode::set_inference_mode(bool enabled) {
  if (enabled) {
    autograd_mode_bitfield |= AutogradMode::INFERENCE_MODE_MASK;
  } else {
    autograd_mode_bitfield &= ~AutogradMode::INFERENCE_MODE_MASK;
  }
}

bool AutogradMode::get_grad_mode() {
  return AutogradMode::get_mode() & AutogradMode::GRAD_MODE_MASK;
}

bool AutogradMode::get_fw_grad_mode() {
  return AutogradMode::get_mode() & AutogradMode::FW_GRAD_MODE_MASK;
}

bool AutogradMode::get_inference_mode() {
  return AutogradMode::get_mode() & AutogradMode::INFERENCE_MODE_MASK;
}

} // namespace c10