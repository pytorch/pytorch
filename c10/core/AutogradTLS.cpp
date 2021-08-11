#include <c10/core/AutogradTLS.h>

namespace c10 {

// By default, both grad modes are enabled and inference mode is disabled
thread_local uint8_t autograd_mode_bitfield =
    AutogradTLS::GRAD_MODE_MASK | AutogradTLS::FW_GRAD_MODE_MASK;

uint8_t AutogradTLS::get_mode() {
  return autograd_mode_bitfield;
}

void AutogradTLS::set_mode(uint8_t flag) {
  autograd_mode_bitfield = flag;
}

void AutogradTLS::set_grad_mode(bool enabled) {
  if (enabled) {
    autograd_mode_bitfield |= AutogradTLS::GRAD_MODE_MASK;
  } else {
    autograd_mode_bitfield &= ~AutogradTLS::GRAD_MODE_MASK;
  }
}

void AutogradTLS::set_fw_grad_mode(bool enabled) {
  if (enabled) {
    autograd_mode_bitfield |= AutogradTLS::FW_GRAD_MODE_MASK;
  } else {
    autograd_mode_bitfield &= ~AutogradTLS::FW_GRAD_MODE_MASK;
  }
}

void AutogradTLS::set_inference_mode(bool enabled) {
  if (enabled) {
    autograd_mode_bitfield |= AutogradTLS::INFERENCE_MODE_MASK;
  } else {
    autograd_mode_bitfield &= ~AutogradTLS::INFERENCE_MODE_MASK;
  }
}

bool AutogradTLS::get_grad_mode() {
  return AutogradTLS::get_mode() & AutogradTLS::GRAD_MODE_MASK;
}

bool AutogradTLS::get_fw_grad_mode() {
  return AutogradTLS::get_mode() & AutogradTLS::FW_GRAD_MODE_MASK;
}

bool AutogradTLS::get_inference_mode() {
  return AutogradTLS::get_mode() & AutogradTLS::INFERENCE_MODE_MASK;
}

} // namespace c10