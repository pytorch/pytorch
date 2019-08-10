namespace torch {
namespace autograd {
namespace amp {
// C++ API should mirror Python API.
struct Amp {
  static bool is_grad_scaling_enabled();
  static void set_grad_scaling_enabled(bool new_enabled);
  static float get_grad_scale();
  static void set_grad_scale(float new_scale);
};
} // namespace torch
} // namespace autograd
} // namespace amp
