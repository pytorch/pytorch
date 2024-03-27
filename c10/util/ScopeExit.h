#pragma once

#include <type_traits>
#include <utility>

namespace c10 {

/**
 * Mostly copied from https://llvm.org/doxygen/ScopeExit_8h_source.html
 */
template <typename Callable>
class scope_exit {
  Callable ExitFunction;
  bool Engaged = true; // False once moved-from or release()d.

 public:
  template <typename Fp>
  // constructor accepting a forwarding reference can hide the
  // move constructor
  // @lint-ignore CLANGTIDY
  explicit scope_exit(Fp&& F) : ExitFunction(std::forward<Fp>(F)) {}

  scope_exit(scope_exit&& Rhs) noexcept
      : ExitFunction(std::move(Rhs.ExitFunction)), Engaged(Rhs.Engaged) {
    Rhs.release();
  }
  scope_exit(const scope_exit&) = delete;
  scope_exit& operator=(scope_exit&&) = delete;
  scope_exit& operator=(const scope_exit&) = delete;

  void release() {
    Engaged = false;
  }

  ~scope_exit() {
    if (Engaged) {
      ExitFunction();
    }
  }
};

// Keeps the callable object that is passed in, and execute it at the
// destruction of the returned object (usually at the scope exit where the
// returned object is kept).
//
// Interface is specified by p0052r2.
template <typename Callable>
scope_exit<typename std::decay<Callable>::type> make_scope_exit(Callable&& F) {
  return scope_exit<typename std::decay<Callable>::type>(
      std::forward<Callable>(F));
}

} // namespace c10
