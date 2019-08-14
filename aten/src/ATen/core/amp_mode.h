#pragma once

#include <c10/macros/Macros.h>

namespace at {

// C++ API should mirror Python API.
// Dunno why CAFFE2_API needs to be here, cargo culting grad_mode.h
// Should the stuff in amp_mode.h and amp_mode.cpp just be moved to ATen/native/Amp[.h,.cpp]?
struct CAFFE2_API AmpMode {
  static bool is_grad_scaling_enabled();
  static void set_grad_scaling_enabled(bool new_enabled);
  static float get_grad_scale();
  static void set_grad_scale(float new_scale);
};

}
