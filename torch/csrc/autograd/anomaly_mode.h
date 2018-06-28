#pragma once

namespace torch { namespace autograd {

struct AnomalyMode {
  static bool is_enabled() {
    return _enabled;
  }
  static void set_enabled(bool enabled) {
    _enabled = enabled;
  }

private:
  static bool _enabled;
};


struct AnomalyMetadata {
  virtual ~AnomalyMetadata(){};
  virtual void store_stack() = 0;
  virtual void print_stack() = 0;
};

}}
