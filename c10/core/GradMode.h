#pragma once

#include <c10/core/impl/ThreadLocalState.h>
#include <c10/macros/Macros.h>

namespace c10 {

struct TORCH_API GradMode {
  // TLS lookup is expensive and this API boils down to a single access on the
  // struct, so if we already have the TLS handy there is a significant
  // performance gain from reusing it and inlining the access. Otherwise, the
  // no argument overload serves as a generic fallback.
  static inline bool is_enabled(const impl::PODLocalState* const tls) {
    return !(tls->GradMode_disabled);
  }

  static bool is_enabled();
  static void set_enabled(bool enabled);
};

// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API AutoGradMode {
  AutoGradMode(bool enabled) : prev_mode(GradMode::is_enabled()) {
    GradMode::set_enabled(enabled);
  }
  ~AutoGradMode() {
    GradMode::set_enabled(prev_mode);
  }
  bool prev_mode;
};

// A RAII, thread local (!) guard that stops future operations from building
// gradients.
struct TORCH_API NoGradGuard : public AutoGradMode {
  NoGradGuard() : AutoGradMode(/*enabled=*/false) {}
};

} // namespace c10
