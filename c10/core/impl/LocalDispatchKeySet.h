#pragma once

#include <c10/core/DispatchKeySet.h>
#include <c10/util/Flags.h>

// TLS management for DispatchKeySet (the "local" DispatchKeySet(s))
//
// This manages two thread-local DispatchKeySets:
//
//  - The included type set, which adds a tensor type for consideration
//    in dispatch.  (For example, you might add ProfilingTensorId to
//    the included type set to turn on profiling on all tensor operations.)
//
//  - The excluded type set, which disqualifies a tensor type from dispatch.
//    (For example, after redispatching on variable, we disqualify
//    VariableTensorId so we don't attempt to handle variable again.)
//    (Exclusion wins over inclusion.)
//
// NB: Originally, I implemented the excluded type set as storing the inverted
// set, but TLS is defined to be zero-initialized, so this doesn't actually work
// (if it's inverted, you want the set to be -1 initialized).

namespace c10 {
namespace impl {

C10_DECLARE_bool(disable_variable_dispatch);

// POD version of LocalDispatchKeySet.  Declared here just so that
// we can put it in the guards.
struct C10_API PODLocalDispatchKeySet {
  uint64_t included_;
  uint64_t excluded_;

  DispatchKeySet included() const {
    return DispatchKeySet(DispatchKeySet::RAW, included_);
  }
  DispatchKeySet excluded() const {
    return DispatchKeySet(DispatchKeySet::RAW, excluded_);
  }

  void set_included(DispatchKeySet x) {
    included_ = x.raw_repr();
  }
  void set_excluded(DispatchKeySet x) {
    excluded_ = x.raw_repr();
  }
};
static_assert(std::is_pod<PODLocalDispatchKeySet>::value, "PODLocalDispatchKeySet must be a POD type.");

struct C10_API LocalDispatchKeySet {
  /* implicit */ LocalDispatchKeySet(PODLocalDispatchKeySet x)
    : included_(x.included()), excluded_(x.excluded()) {}
  DispatchKeySet included_;
  DispatchKeySet excluded_;
};

C10_API LocalDispatchKeySet tls_local_dispatch_key_set();

// RAII API for manipulating the thread-local dispatch state.

class C10_API IncludeDispatchKeyGuard {
public:
  IncludeDispatchKeyGuard(DispatchKey);
  IncludeDispatchKeyGuard(const IncludeDispatchKeyGuard&) = delete;
  IncludeDispatchKeyGuard operator=(const IncludeDispatchKeyGuard&) = delete;
  IncludeDispatchKeyGuard(IncludeDispatchKeyGuard&&) = delete;
  IncludeDispatchKeyGuard operator=(IncludeDispatchKeyGuard&&) = delete;
  ~IncludeDispatchKeyGuard();
private:
  // A little micro-optimization to save us from tls_get_addr call
  // on destruction
  PODLocalDispatchKeySet* tls_;
  DispatchKey id_;
  bool prev_state_;
};

class C10_API ExcludeDispatchKeyGuard {
public:
  ExcludeDispatchKeyGuard(DispatchKey);
  ExcludeDispatchKeyGuard(const ExcludeDispatchKeyGuard&) = delete;
  ExcludeDispatchKeyGuard operator=(const ExcludeDispatchKeyGuard&) = delete;
  ExcludeDispatchKeyGuard(ExcludeDispatchKeyGuard&&) = delete;
  ExcludeDispatchKeyGuard operator=(ExcludeDispatchKeyGuard&&) = delete;
  ~ExcludeDispatchKeyGuard();
private:
  // A little micro-optimization to save us from tls_get_addr call
  // on destruction
  PODLocalDispatchKeySet* tls_;
  DispatchKey id_;
  bool prev_state_;
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
// getter and setter will do a tls_getaddr lookup (the RAII struct only needs one!)

C10_API bool tls_is_dispatch_key_excluded(DispatchKey x);
C10_API void tls_set_dispatch_key_excluded(DispatchKey x, bool desired_state);
C10_API bool tls_is_dispatch_key_included(DispatchKey x);
C10_API void tls_set_dispatch_key_included(DispatchKey x, bool desired_state);

}} // namespace c10::impl
