#pragma once

#include <memory>

#include <ATen/core/TensorBody.h>
#include <c10/core/TensorImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/strong_type.h>

namespace torch {
namespace profiler {
namespace impl {

// Identity is a complex concept in PyTorch. A Tensor might not have a
// an associated storage, multiple Tensors might share the same underlying
// storage, the storage of a Tensor might change over time, etc.
//
// For the purpose of profiling we're mostly interested in data flow
// analysis. As a result, we can take an expansive view of identity:
// Tensors share an ID if they share a TensorImpl or storage data.
//
// This identity equality is transitive; If Tensors T0 and T1 share a storage
// S0 and T1 later points to a different storage S1 then all Tensors which
// point to either S0 or S1 are considered to have the same identity. (Since
// profiler cannot reason beyond that.)
//
// The profiler will handle lifetime analysis to ensure that identities do
// not run afoul of the ABA problem. This does, however, mean that identities
// can only be assigned when memory profiling is enabled.
using TensorID = strong::type<size_t, struct TensorID_, strong::regular>;

// Uniquely identifies an allocation. (Generally a StorageImpl's data ptr.)
using AllocationID = strong::type<
    size_t,
    struct StorageID_,
    strong::ordered,
    strong::regular,
    strong::hashable>;

// We use a Tensor's TensorImpl adress and StorageImpl data start to build the
// data flow graph. We do not hold an owning reference so we wrap them in strong
// types to prevent direct access.
using TensorImplAddress = strong::type<
    const c10::TensorImpl*,
    struct TensorImplAddress_,
    strong::regular,
    strong::hashable,
    strong::boolean>;

using StorageImplData = strong::type<
    const void*,
    struct StorageImplData_,
    strong::regular,
    strong::hashable,
    strong::boolean>;

// ============================================================================
// == weak_intrusive_ptr and the ABA problem for TensorImpl* ==================
// ============================================================================
// Tracking `TensorImpl`s is an important part of identity tracking, because
// a Tensor might change storage; however when it does we want to retain the
// fact that the old and new storage belong to the same logical Tensor. We
// cannot take an owning reference to the Tensor because that would change
// program semantics by extending the lifetime of the Tensor. However if we
// store a raw TensorImpl* pointer the TensorImpl might be deleted and a new
// TensorImpl might be created that reuses the address. (ABA problem)
//
// Fortunately, there is a feature of `c10::intrusive_ptr` that we can use to
// prevent address reuse for the duration of profiling: the weak intrusive ptr.
// When a Tensor's refcount reaches zero but there are outstanding weak
// references (`weakcount_ > 0`) it will free the underlying managed resources
// by calling `target_->release_resources()`, but it will not call `delete`.
// (Instead, `delete` is called when the last weak reference is destroyed.)
// This means that we can safely use address identity to track `TensorImpls`.
class WeakTensor {
 public:
  explicit WeakTensor(const at::Tensor& t) : weak_self_(t.getIntrusivePtr()) {}

  auto get() const {
    return TensorImplAddress{weak_self_._unsafe_get_target()};
  }

 private:
  c10::weak_intrusive_ptr<c10::TensorImpl> weak_self_;
};

struct Result;

void calculateUniqueTensorIDs(
    std::vector<std::shared_ptr<Result>>& sorted_results);

} // namespace impl
} // namespace profiler
} // namespace torch
