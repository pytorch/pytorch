#pragma once

#include <c10/macros/Export.h>
#include <atomic>

namespace c10::impl {

// This TLS controls whether or not we permanently associate PyObject
// with Tensor the first time it is allocated.  When hermetic PyObject
// TLS is enabled (state is true), we DO NOT save PyObjects to Tensor,
// meaning you get a distinct PyObject whenever you execute the code in
// question.
struct C10_API HermeticPyObjectTLS {
  static void set_state(bool state);
  static bool get_state() {
    // Hypothetical fastpath if torchdeploy/multipy isn't used.  Per
    // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2055r0.pdf
    // this qualifies relaxed access because it is a single-location data
    // structure (only the boolean here).
    //
    // Forgetting about data races for a moment, is there a logical race?
    //
    //  - Boolean only ever transitions from false to true.  So the
    //    critical situation is when one interpreter is already running
    //    when a second interpreter switches haveState from false to true.
    //
    //  - The first interpreter is indifferent whether or not it sees
    //    hasState true/false; obviously false works (this is what the
    //    interpreter was previously using; more directly, the interpreter
    //    calls into itself as the handler, so being hermetic is not
    //    required), and true simply means serviced python operator calls will
    //    be hermetic; in these cases it is expected to be functionally
    //    equivalent.
    //
    //  - The second interpreter MUST see hasState true (as its requests will
    //    be forwarded to the first interpreter), but it is assumed that there
    //    is a synchronization between the interpreter initialization, and
    //    when we actually perform operations, so it is guaranteed to see
    //    hasState true.
    //
    // QED.
    //
    // This fastpath is currently disabled so that we can more easily test that
    // hermetic mode works correctly even on stock build of PyTorch.
    if (false && !haveState_.load(std::memory_order_relaxed))
      return false;
    return get_tls_state();
  }
  // Call this from the multipy/torchdeploy top level
  static void init_state();

 private:
  // This only flipped once from false to true during torchdeploy/multipy
  // initialization, and never again.
  static std::atomic<bool> haveState_;
  static bool get_tls_state();
};

} // namespace c10::impl
