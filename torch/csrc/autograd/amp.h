namespace torch {
namespace autograd {
namespace amp {
bool getGradScalingEnabled();
void setGradScalingEnabled(bool new_enabled);
float getGradScale();
void setGradScale(float new_scale);
} // namespace torch
} // namespace autograd
} // namespace amp
