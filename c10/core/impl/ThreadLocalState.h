#pragma once

#include <c10/core/DispatchKeySet.h>
#include <c10/macros/Macros.h>

namespace c10 {
namespace impl {

// NB: POD, must be zero initialized!
// Note [TLS Initialization]
// We would prefer fields to be initialized with non-zero state
// e.g. BackendSelect and ADInplaceOrView in included set.  But certain Windows
// compiler (e.g the one used in ARVR tests) only allow TLS to be
// zero-initialized. To preserve the invariant that raw TLS storage of the
// default state is zero, we obtain the actual include keyset by XORing
// the raw bits (included / excluded) with an XOR mask value.
//
// Moreover, PODLocalState is a simple data container. Logic for interpreting
// or updating fields should be owned by higher level containers which operate
// on a PODLocalState* pointer.
struct C10_API PODLocalState {
 public:
  // DispatchKeySet
  uint64_t included_;
  uint64_t excluded_;

  // GradMode
  bool GradMode_disabled;
};

static_assert(
    std::is_pod<PODLocalState>::value,
    "PODLocalState must be a POD type.");

#if defined(_MSC_VER) || defined(C10_ANDROID)
C10_API PODLocalState* _get_thread_local_state();
#else // defined(_MSC_VER) || defined(C10_ANDROID)
extern C10_API thread_local PODLocalState raw_thread_local_state;

inline C10_API PODLocalState* _get_thread_local_state() {
  return &raw_thread_local_state;
}
#endif // defined(_MSC_VER) || defined(C10_ANDROID)

} // namespace impl
} // namespace c10
