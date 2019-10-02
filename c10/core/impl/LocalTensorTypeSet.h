#pragma once

#include <c10/core/TensorTypeSet.h>

// TLS management for TensorTypeSet (the "local" TensorTypeSet(s))
//
// This manages two thread-local TensorTypeSets:
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

// POD version of LocalTensorTypeSet.  Declared here just so that
// we can put it in the guards.
struct C10_API PODLocalTensorTypeSet {
  uint64_t included_;
  uint64_t excluded_;

  TensorTypeSet included() const {
    return TensorTypeSet(TensorTypeSet::RAW, included_);
  }
  TensorTypeSet excluded() const {
    return TensorTypeSet(TensorTypeSet::RAW, excluded_);
  }

  void set_included(TensorTypeSet x) {
    included_ = x.raw_repr();
  }
  void set_excluded(TensorTypeSet x) {
    excluded_ = x.raw_repr();
  }
};
static_assert(std::is_pod<PODLocalTensorTypeSet>::value, "PODLocalTensorTypeSet must be a POD type.");

struct C10_API LocalTensorTypeSet {
  /* implicit */ LocalTensorTypeSet(PODLocalTensorTypeSet x)
    : included_(x.included()), excluded_(x.excluded()) {}
  TensorTypeSet included_;
  TensorTypeSet excluded_;
};

C10_API LocalTensorTypeSet tls_local_tensor_type_set();

class C10_API IncludeTensorTypeIdGuard {
public:
  IncludeTensorTypeIdGuard(TensorTypeId);
  ~IncludeTensorTypeIdGuard();
private:
  // A little micro-optimization to save us from tls_get_addr call
  // on destruction
  PODLocalTensorTypeSet* tls_;
  TensorTypeId id_;
  bool prev_state_;
};

class C10_API ExcludeTensorTypeIdGuard {
public:
  ExcludeTensorTypeIdGuard(TensorTypeId);
  ~ExcludeTensorTypeIdGuard();
private:
  // A little micro-optimization to save us from tls_get_addr call
  // on destruction
  PODLocalTensorTypeSet* tls_;
  TensorTypeId id_;
  bool prev_state_;
};

}} // namespace c10::impl
