#pragma once

#include <c10/util/intrusive_ptr.h>
#include <ATen/core/jit_type.h>

namespace c10 {

using worker_id_t = int16_t;

// This abstract class contains only user-facing APIs, and will be shared
// between jit and distributed to implement TorchScript support.
class C10_EXPORT RRef : public c10::intrusive_ptr_target {
 private:
  c10::intrusive_ptr<RRef> intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                           // from a raw `this` pointer
                                           // so we need to bump the refcount
                                           // to account for this ownership
    return c10::intrusive_ptr<RRef>::reclaim(this);
  }

 public:
  RRef() = default;
  // RRef is made NOT copyable NOT movable to prevent messing up reference
  // counting.
  RRef(const RRef& other) = delete;
  RRef(RRef&& other) = delete;
  RRef& operator=(RRef&& other) = delete;

  virtual ~RRef() = default;

  // returns the worker id of the owner
  virtual worker_id_t owner() const = 0;

  // Returns true if this is the ``OwnerRRef``
  virtual bool isOwner() const = 0;

  // virtual TypePtr type() const = 0;
};

}