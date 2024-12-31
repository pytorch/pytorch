#pragma once

#include <c10/core/DispatchKeySet.h>
#include <c10/macros/Export.h>

// TLS management for DispatchKeySet (the "local" DispatchKeySet(s))
//
// This manages two thread-local DispatchKeySets:
//
//  - The included type set, which adds a tensor type for consideration
//    in dispatch.  (For example, you might add Profiling to
//    the included type set to turn on profiling on all tensor operations.)
//
//  - The excluded type set, which disqualifies a tensor type from dispatch.
//    (For example, after redispatching on variable, we disqualify
//    Autograd so we don't attempt to handle variable again.)
//    (Exclusion wins over inclusion.)
//
// NB: Originally, I implemented the excluded type set as storing the inverted
// set, but TLS is defined to be zero-initialized, so this doesn't actually work
// (if it's inverted, you want the set to be -1 initialized).

namespace c10::impl {

// POD version of LocalDispatchKeySet.  Declared here just so that
// we can put it in the guards.
// This struct encapsulates special handling for TLS initialization
// in set_included()/included() API so that they reflect the truth.
// If you want to create PODLocalDispatchKeySet with non-zero state,
// use set_included() instead of default constructor.
struct C10_API PODLocalDispatchKeySet {
  uint64_t included_;
  uint64_t excluded_;

  // See Note [TLS Initialization]
  DispatchKeySet included() const {
    return DispatchKeySet(DispatchKeySet::RAW, included_) ^
        c10::default_included_set;
  }
  DispatchKeySet excluded() const {
    return DispatchKeySet(DispatchKeySet::RAW, excluded_) ^
        c10::default_excluded_set;
  }

  void set_included(DispatchKeySet x) {
    included_ = (x ^ c10::default_included_set).raw_repr();
  }
  void set_excluded(DispatchKeySet x) {
    excluded_ = (x ^ c10::default_excluded_set).raw_repr();
  }
};
static_assert(
    std::is_trivial_v<PODLocalDispatchKeySet>,
    "PODLocalDispatchKeySet must be a POD type.");

struct C10_API LocalDispatchKeySet {
  /* implicit */ LocalDispatchKeySet(PODLocalDispatchKeySet x)
      : included_(x.included()), excluded_(x.excluded()) {}
  DispatchKeySet included_;
  DispatchKeySet excluded_;
};

// thread_local variables cannot be C10_API on Windows.
// Inlining this seems to break AutoDispatchBelowAutograd on Android.
#if defined(_MSC_VER) || defined(C10_ANDROID) || defined(C10_IPHONE)
C10_API LocalDispatchKeySet tls_local_dispatch_key_set();
#else // defined(_MSC_VER) || defined(C10_ANDROID) || defined(C10_IPHONE)
extern C10_API thread_local PODLocalDispatchKeySet raw_local_dispatch_key_set;

inline C10_API LocalDispatchKeySet tls_local_dispatch_key_set() {
  // Don't let people fiddle with the thread_local directly just
  // because they include this header.
  return raw_local_dispatch_key_set;
}
#endif // defined(_MSC_VER) || defined(C10_ANDROID) || defined(C10_IPHONE)

// Internal, use ThreadLocalStateGuard
C10_API void _force_tls_local_dispatch_key_set(LocalDispatchKeySet key_set);

// RAII API for manipulating the thread-local dispatch state.

class C10_API IncludeDispatchKeyGuard {
 public:
  IncludeDispatchKeyGuard(DispatchKeySet);
  IncludeDispatchKeyGuard(DispatchKey k)
      : IncludeDispatchKeyGuard(DispatchKeySet(k)) {}
  IncludeDispatchKeyGuard(const IncludeDispatchKeyGuard&) = delete;
  IncludeDispatchKeyGuard operator=(const IncludeDispatchKeyGuard&) = delete;
  IncludeDispatchKeyGuard(IncludeDispatchKeyGuard&&) = delete;
  IncludeDispatchKeyGuard operator=(IncludeDispatchKeyGuard&&) = delete;
  ~IncludeDispatchKeyGuard();

 private:
  // A little micro-optimization to save us from tls_get_addr call
  // on destruction
  PODLocalDispatchKeySet* tls_;
  DispatchKeySet include_;
};

class C10_API ExcludeDispatchKeyGuard {
 public:
  ExcludeDispatchKeyGuard(DispatchKeySet);
  ExcludeDispatchKeyGuard(DispatchKey k)
      : ExcludeDispatchKeyGuard(DispatchKeySet(k)) {}
  ExcludeDispatchKeyGuard(const ExcludeDispatchKeyGuard&) = delete;
  ExcludeDispatchKeyGuard operator=(const ExcludeDispatchKeyGuard&) = delete;
  ExcludeDispatchKeyGuard(ExcludeDispatchKeyGuard&&) = delete;
  ExcludeDispatchKeyGuard operator=(ExcludeDispatchKeyGuard&&) = delete;
  ~ExcludeDispatchKeyGuard();

 private:
  // A little micro-optimization to save us from tls_get_addr call
  // on destruction
  PODLocalDispatchKeySet* tls_;
  DispatchKeySet exclude_;
};

struct C10_API ForceDispatchKeyGuard {
 public:
  ForceDispatchKeyGuard()
      : saved_keyset_(c10::impl::tls_local_dispatch_key_set()) {}
  ForceDispatchKeyGuard(c10::impl::LocalDispatchKeySet key_set)
      : ForceDispatchKeyGuard() {
    c10::impl::_force_tls_local_dispatch_key_set(key_set);
  }
  ForceDispatchKeyGuard(
      c10::DispatchKeySet include,
      c10::DispatchKeySet exclude)
      : ForceDispatchKeyGuard() {
    auto updated_set = saved_keyset_;
    updated_set.included_ = include;
    updated_set.excluded_ = exclude;
    c10::impl::_force_tls_local_dispatch_key_set(updated_set);
  }

  ForceDispatchKeyGuard(ForceDispatchKeyGuard&&) noexcept = delete;
  ForceDispatchKeyGuard(const ForceDispatchKeyGuard&) = delete;
  ForceDispatchKeyGuard& operator=(const ForceDispatchKeyGuard&) = delete;
  ForceDispatchKeyGuard& operator=(ForceDispatchKeyGuard&&) = delete;
  ~ForceDispatchKeyGuard() {
    c10::impl::_force_tls_local_dispatch_key_set(saved_keyset_);
  }

 private:
  c10::impl::LocalDispatchKeySet saved_keyset_;
};

// Non-RAII API for manipulating the thread-local dispatch state.
// Please prefer the RAII API.  The non-RAII API may be useful when
// the included/excluded state of a given DispatchKey must span
// many calls from the Python to the C++, so you cannot conveniently
// use an RAII guard.
//
// Example use case:  a Python context manager that includes a certain
// DispatchKey, to ensure ops running under the context manager dispatch
// through that DispatchKey's registered overrides.
//
// The non-RAII API is less efficient than the RAII guards because both the
// getter and setter will do a tls_getaddr lookup (the RAII struct only needs
// one!)

C10_API bool tls_is_dispatch_key_excluded(DispatchKey x);
C10_API void tls_set_dispatch_key_excluded(DispatchKey x, bool desired_state);
C10_API bool tls_is_dispatch_key_included(DispatchKey x);
C10_API void tls_set_dispatch_key_included(DispatchKey x, bool desired_state);
C10_API bool tls_is_dispatch_keyset_excluded(DispatchKeySet ks);
C10_API bool tls_is_dispatch_keyset_included(DispatchKeySet ks);

} // namespace c10::impl
