#pragma once

#include <c10/macros/Macros.h>

#include <cstdint>

namespace c10 {

// Structure used to pack all the thread local boolean
// flags used by autograd
struct TORCH_API AutogradTLS {
  static uint8_t get_mode();
  static void set_mode(uint8_t flag);
  static void set_grad_mode(bool enabled);
  static void set_fw_grad_mode(bool enabled);
  static void set_inference_mode(bool enabled);
  static bool get_grad_mode();
  static bool get_fw_grad_mode();
  static bool get_inference_mode();

  static const uint8_t GRAD_MODE_MASK = 1;
  static const uint8_t INFERENCE_MODE_MASK = 2;
  static const uint8_t FW_GRAD_MODE_MASK = 4;
};

} // namespace c10
