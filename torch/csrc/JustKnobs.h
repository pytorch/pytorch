#pragma once

#include <cstdint>
#include <string_view>

/*
    JustKnobs is a lightweight configuration system. It supports
    re-configuring values in real-time (i.e., without a job needing
    to restart).

    Below, we have a
    1. Meta-only implementation. Delegates to our underlying library.
    2. Default OSS implementation. Always returns the default.

    Please see the corresponding test file for basic usage and concepts.
*/

namespace torch::jk {

#ifdef FBCODE_CAFFE2

#include <justknobs/JustKnobs.h>

class BooleanKnob {
 public:
  explicit BooleanKnob(std::string_view name) : real_knob_(name) {}

  template <typename... Args>
  bool operator()(bool default_value, Args&&... args) const {
    try {
      return real_knob_(std::forward<Args>(args)...);
    } catch (...) {
      return default_value;
    }
  }

 private:
  ::facebook::jk::BooleanKnob real_knob_;
};

class IntegerKnob {
 public:
  explicit IntegerKnob(std::string_view name) : real_knob_(name) {}

  template <typename... Args>
  int64_t operator()(int64_t default_value, Args&&... args) const {
    try {
      return real_knob_(std::forward<Args>(args)...);
    } catch (...) {
      return default_value;
    }
  }

 private:
  ::facebook::jk::IntegerKnob real_knob_;
};

#else

class BooleanKnob {
 public:
  explicit BooleanKnob(std::string_view name) {
    (void)name;
  }

  template <typename... Args>
  bool operator()(bool default_value, Args&&...) const {
    return default_value;
  }
};

class IntegerKnob {
 public:
  explicit IntegerKnob(std::string_view name) {
    (void)name;
  }

  template <typename... Args>
  int64_t operator()(int64_t default_value, Args&&...) const {
    return default_value;
  }
};

#endif

} // namespace torch::jk
