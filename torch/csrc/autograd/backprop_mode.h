#pragma once

namespace torch { namespace autograd {

struct BackpropMode {
  static bool is_enabled() {
    return _enabled;
  }
  static void set_enabled(bool enabled) {
    _enabled = enabled;
  }
private:
  static thread_local bool _enabled;
};

struct AutoBackpropMode {
  AutoBackpropMode(bool enabled) : prev_mode(BackpropMode::is_enabled()) {
    BackpropMode::set_enabled(enabled);
  }
  ~AutoBackpropMode() {
    BackpropMode::set_enabled(prev_mode);
  }
  bool prev_mode;
};

}}
