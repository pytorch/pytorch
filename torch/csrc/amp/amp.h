namespace torch { namespace autograd { namespace amp {
namespace {
  bool enabled = false;
  float loss_scale = 4.0;
}

  inline bool is_enabled() { return enabled; }
  inline void set_enabled(bool new_enabled) { enabled = new_enabled; }
  inline float get_loss_scale() { return loss_scale; }
  inline void set_loss_scale(float new_scale) { loss_scale = new_scale; }
} } }
