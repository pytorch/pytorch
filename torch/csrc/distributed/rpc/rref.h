#pragma once

#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace rpc {

// This abstract class contains only user-facing APIs, and will be shared
// between jit and distributed to implement TorchScript support.
class RRef {
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
};

} // namespace rpc
} // namespace distributed
} // namespace torch
