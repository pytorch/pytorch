#pragma once

#include <c10/util/intrusive_ptr.h>
#include <ATen/core/jit_type_base.h>

namespace c10 {

struct Type;
using worker_id_t = int16_t;

// This abstract class contains only user-facing APIs, and will be shared
// between jit and distributed to implement TorchScript support.
class C10_EXPORT RRefInterface : public c10::intrusive_ptr_target {
 public:
  RRefInterface() = default;
  // RRef is made NOT copyable NOT movable to prevent messing up reference
  // counting.
  RRefInterface(const RRefInterface& other) = delete;
  RRefInterface(RRefInterface&& other) = delete;
  RRefInterface& operator=(const RRefInterface& other) = delete;
  RRefInterface& operator=(RRefInterface&& other) = delete;

  ~RRefInterface() override = default;

  // returns the worker id of the owner
  virtual worker_id_t owner() const = 0;

  // returns the worker name of the owner
  virtual std::string ownerName() const = 0;

  // Returns true if this is the ``OwnerRRef``
  virtual bool isOwner() const = 0;

  // Returns true if this is an ``OwnerRRef`` or if this ``UserRRef`` has been
  // confirmed by its owner.
  virtual bool confirmedByOwner() const = 0;

  virtual const TypePtr type() const = 0;
};

}
