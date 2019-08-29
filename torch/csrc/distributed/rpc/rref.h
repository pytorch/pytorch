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

  RRefForkData(worker_id_t ownerId,
               const RRefId& rrefId_,
               const ForkId& forkId_);

  static RRefForkData fromIValue(const at::IValue&&);

  const worker_id_t ownerId_;
  const RRefId rrefId_;
  const ForkId forkId_;
};

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
  const ForkId& forkId() const;
  IValue fork() const;

  virtual bool isOwner() const = 0;
  virtual void setValue(IValue&& value) = 0;
  virtual IValue getValue() const = 0;
  virtual IValue toHere() = 0;

 protected:
  RRef(worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId);

  const worker_id_t ownerId_;
  const RRefId rrefId_;
  // If this is the owner, forkId_ == rrefId_.
  const ForkId forkId_;
};

class UserRRef final: public RRef {
 public:
  bool isOwner() const override;
  IValue getValue() const override;
  void setValue(IValue&& value) override;
  IValue toHere() override;

  ~UserRRef() override;
 private:
  friend class RRefContext;

  UserRRef(worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId);
};

// Keep the template only on the derived class because ``RRefContext`` needs to
// erase the type on ``RRef`` and keep them in one map.
template <typename T>
class OwnerRRef final: public RRef {
 public:
  bool isOwner() const override {
    return true;
  }

  IValue getValue() const override {
    if(std::is_same<T, IValue>::value) {
      // TODO: use callback to make this non-blocking
      std::unique_lock<std::mutex> lock(mutex_);
      valueCV_.wait(lock, [this]{return value_.has_value();});
      return value_.value();
    } else {
      AT_ERROR("Trying to store an IValue in incompatible RRef[T].");
    }
  }

  void setValue(IValue&& value) override {
    if(std::is_same<T, IValue>::value) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        value_ = std::move(value);
      }
      valueCV_.notify_all();
    } else {
      AT_ERROR("Trying to store an IValue in incompatible RRef[T].");
    }
  }

  IValue toHere() override {
    AT_ERROR("OwnerRRef does not support toHere(), use getValue() instead.");
  }

  // TODO: add setValue(py::object) and getPyObj() for Python UDF

 private:
  friend class RRefContext;

  OwnerRRef(worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId)
      : OwnerRRef(ownerId, rrefId, forkId, {}) {}

  OwnerRRef(OwnerRRef<T>&& other) noexcept
      : OwnerRRef(other.owner(),
                  other.id(),
                  other.forkId(),
                  std::move(other.value_)) {}

  OwnerRRef(
      worker_id_t ownerId,
      const RRefId& rrefId,
      const ForkId& forkId,
      c10::optional<T> value)
      : RRef(ownerId, rrefId, forkId) {
    AT_ASSERT(forkId_ == rrefId_,
        "Owner RRef's fork ID should be the same as its rref Id");

    value_ = std::move(value);
  }

  c10::optional<T> value_;
  mutable std::mutex mutex_;
  mutable std::condition_variable valueCV_;
};


} // namespace rpc
} // namespace distributed
} // namespace torch
