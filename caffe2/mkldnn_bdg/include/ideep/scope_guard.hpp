#ifndef _SCOPE_GUARD_HPP_
#define _SCOPE_GUARD_HPP_

#include <utility>

namespace ideep {
namespace utils {
class sg_impl {
public:
  sg_impl() : armed_(true) {}

  sg_impl(sg_impl&& movable) : armed_(movable.armed_) {
    movable.armed_ = false;
  }
  
  void disarm() {
    armed_ = false;
  }
protected:
  ~sg_impl() = default;
  bool armed_;
};

template <typename F>
class scope_guard : public sg_impl {
public:
  scope_guard() = delete;
  scope_guard(const scope_guard&) = delete;

  scope_guard(F func) noexcept : sg_impl(), func_(std::move(func)) {
  }

  scope_guard(scope_guard&& movable) noexcept
    :sg_impl(std::move(movable)), func_(std::move(movable.func_)) {
  }

  ~scope_guard() noexcept {
    if (armed_) {
      try {
        func_();
      } catch (...) {}
    }
  }
  scope_guard& operator =(const scope_guard&) = delete;
private:
  F func_;
};

template <typename F> scope_guard<F> make_guard(F func) {
  return scope_guard<F>(std::move(func));
}
}
}
#endif
