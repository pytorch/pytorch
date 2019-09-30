#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/types.h>

#include <atomic>

namespace torch {
namespace distributed {
namespace rpc {

class RRef;
class RRefContext;
class UserRRef;

// Represents fork of an RRef to be sent over the wire.
//
// In order to preserve correctness of reference counting, each RRefForkData
// **MUST** be deserialized into a RRef. This means that if RRefForkData is to
// be transferred across the network, we need the guarantee that the message
// will *eventually* get to the peer,  and that the peer will create a RRef out
// of it. Therefore, no constructor of RRefForkData is exposed, and
// applications should never directly use RRefForkData. All construction are
// done within ``RRef`` and ``RRefContext``.
struct RRefForkData {
  at::IValue toIValue() const;

 private:
  friend class RRef;
  friend class RRefContext;
  friend class UserRRef;

  RRefForkData(
      worker_id_t ownerId,
      const RRefId& rrefId_,
      const ForkId& forkId_);

  static RRefForkData fromIValue(const at::IValue&);

  const worker_id_t ownerId_;
  const RRefId rrefId_;
  const ForkId forkId_;
};

static_assert(
    C10_IS_TRIVIALLY_COPYABLE(RRefForkData),
    "RRefForkData must be trivially copyable");

// TODO: make RRef an IValue, and edit createStackForSchema accordingly
class RRef {
 public:
  // RRef is made NOT copyable NOT movable to prevent messing up reference
  // counting
  RRef(const RRef& other) = delete;
  RRef(RRef&& other) = delete;

  virtual ~RRef() = default;

  worker_id_t owner() const;
  const RRefId& id() const;
  IValue fork() const;

  virtual bool isOwner() const = 0;
  virtual IValue toHere() = 0;

 protected:
  RRef(worker_id_t ownerId, const RRefId& rrefId);

  const worker_id_t ownerId_;
  const RRefId rrefId_;
};

class UserRRef final : public RRef {
 public:
  const ForkId& forkId() const;
  bool isOwner() const override;
  IValue toHere() override;

  ~UserRRef() override;

 private:
  friend class RRefContext;

  UserRRef(worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId);

  const ForkId forkId_;
};

// Keep the template only on the derived class because ``RRefContext`` needs to
// erase the type on ``RRef`` and keep them in one map.
template <typename T>
class OwnerRRef final : public RRef {
 public:
  bool isOwner() const override {
    return true;
  }

  T getValue() const {
    // TODO: use callback to make this non-blocking
    std::unique_lock<std::mutex> lock(mutex_);
    valueCV_.wait(lock, [this] { return value_.has_value(); });
    return value_.value();
  }

  void setValue(T&& value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      value_ = std::move(value);
    }
    valueCV_.notify_all();
  }

  IValue toHere() override {
    AT_ERROR("OwnerRRef does not support toHere(), use getValue() instead.");
  }

 private:
  friend class RRefContext;

  OwnerRRef(worker_id_t ownerId, const RRefId& rrefId)
      : OwnerRRef(ownerId, rrefId, {}) {}

  OwnerRRef(OwnerRRef<T>&& other) noexcept
      : OwnerRRef(other.owner(), other.id(), std::move(other.value_)) {}

  OwnerRRef(worker_id_t ownerId, const RRefId& rrefId, c10::optional<T> value)
      : RRef(ownerId, rrefId) {
    value_ = std::move(value);
  }

  c10::optional<T> value_;
  mutable std::mutex mutex_;
  mutable std::condition_variable valueCV_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
