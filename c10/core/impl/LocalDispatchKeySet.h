#pragma once

#include <type_traits>

#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/ThreadLocalState.h>
#include <c10/macros/Macros.h>
#include <c10/util/Flags.h>

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

namespace c10 {
namespace impl {

// Helper class to handle conversion between the TLS zero initialized
// convention and the "normal" in memory layout.
class C10_API LocalDispatchKeySetWrapper {
 public:
  explicit LocalDispatchKeySetWrapper(PODLocalState* tls) : tls_(tls) {}

  // See Note [TLS Initialization]
  DispatchKeySet included() const {
    return DispatchKeySet(DispatchKeySet::RAW, tls_->included_) ^
        c10::default_included_set;
  }
  DispatchKeySet excluded() const {
    return DispatchKeySet(DispatchKeySet::RAW, tls_->excluded_) ^
        c10::default_excluded_set;
  }

  void set_included(DispatchKeySet x) {
    tls_->included_ = (x ^ c10::default_included_set).raw_repr();
  }
  void set_excluded(DispatchKeySet x) {
    tls_->excluded_ = (x ^ c10::default_excluded_set).raw_repr();
  }

 private:
  PODLocalState* tls_;
};

struct C10_API LocalDispatchKeySet {
  LocalDispatchKeySet(LocalDispatchKeySetWrapper x)
      : included_(x.included()), excluded_(x.excluded()) {}
  DispatchKeySet included_;
  DispatchKeySet excluded_;
};

inline C10_API LocalDispatchKeySet snapshot_tls_keyset() {
  return LocalDispatchKeySet(
      LocalDispatchKeySetWrapper(_get_thread_local_state()));
}

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
  LocalDispatchKeySetWrapper tls_wrapper_;
  DispatchKeySet include_;
};

template <uint64_t exclude, bool has_overlap>
class ExcludeDispatchKeyGuard {
 public:
  // If our exclude set does not overlap with c10::default_excluded_set, we can
  // skip some bookkeeping. (And we know at compile time if this is the case.)
  // Key exclusion tends to be on the hot path, so it's worth it to bypass the
  // unnecessary XORs if possible. We force `has_overlap` to be explicitly
  // declared so that changes to the default excluded set do not silently knock
  // us off the hot path.
  static_assert(
      has_overlap == (bool)(exclude & c10::default_excluded_set.raw_repr()),
      "Declared `has_overlap` does not match computed value.");

  ExcludeDispatchKeyGuard(const ExcludeDispatchKeyGuard&) = delete;
  ExcludeDispatchKeyGuard operator=(const ExcludeDispatchKeyGuard&) = delete;
  ExcludeDispatchKeyGuard(ExcludeDispatchKeyGuard&&) = delete;
  ExcludeDispatchKeyGuard operator=(ExcludeDispatchKeyGuard&&) = delete;

  explicit ExcludeDispatchKeyGuard(PODLocalState* tls) : tls_(tls) {
    if (has_overlap) {
      LocalDispatchKeySetWrapper wrapper{tls_};
      auto current_excluded = wrapper.excluded();
      auto exclude_set = DispatchKeySet(DispatchKeySet::RAW, exclude);
      delta_ = (exclude_set - current_excluded).raw_repr();
      wrapper.set_excluded(current_excluded | exclude_set);

    } else {
      // Fast path
      delta_ = exclude & ~(tls_->excluded_);
      tls_->excluded_ |= exclude;
    }
  }

  // Certain key guards (such as `AutoDispatchBelowADInplaceOrView`) are part
  // of the public API, so we have to allow instantiation without exposing
  // the implementation detail of `_get_thread_local_state()`.
  ExcludeDispatchKeyGuard()
      : ExcludeDispatchKeyGuard(_get_thread_local_state()) {}

  ~ExcludeDispatchKeyGuard() {
    if (has_overlap) {
      LocalDispatchKeySetWrapper wrapper{tls_};
      auto current = wrapper.excluded();
      auto delta = DispatchKeySet(DispatchKeySet::RAW, delta_);
      wrapper.set_excluded(current - delta);

    } else {
      // Fast path
      tls_->excluded_ &= ~delta_;
    }
  };

 private:
  // A little micro-optimization to save us from tls_get_addr call
  // on destruction
  PODLocalState* tls_;
  uint64_t delta_;
};

template <DispatchKey k, bool has_overlap>
class ExcludeSingleDispatchKeyGuard {
  static constexpr auto k_set = DispatchKeySet(k);
  ExcludeDispatchKeyGuard<k_set.raw_repr(), has_overlap> guard_;
};

#define SPECIALIZE_EXCLUDE_GUARD(guard_name, k, has_overlap)              \
  class TORCH_API guard_name                                              \
      : c10::impl::ExcludeDispatchKeyGuard<k.raw_repr(), has_overlap> {   \
    using c10::impl::ExcludeDispatchKeyGuard<k.raw_repr(), has_overlap>:: \
        ExcludeDispatchKeyGuard;                                          \
  }

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

} // namespace impl
} // namespace c10
