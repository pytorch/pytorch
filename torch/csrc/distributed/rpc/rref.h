#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/types.h>

#include <atomic>

namespace torch {
namespace distributed {
namespace rpc {


class RRefContext;
class RRef;

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

  ~RRef();

  worker_id_t owner() const;
  const RRefId& id() const;
  const ForkId& forkId() const;
  bool isOwner() const;
  IValue toHere();
  IValue fork() const;

  virtual void setValue(IValue&& value) = 0;
  virtual IValue getValue() = 0;

  // TODO: add setValue(py::object) and getPyObj() for Python UDF

 protected:
  friend class RRefContext;

  RRef(worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId);

  const worker_id_t ownerId_;
  const RRefId rrefId_;
  // If this is the owner, forkId_ == rrefId_.
  const ForkId forkId_;
  c10::optional<std::unordered_set<ForkId, ForkId::Hash>> children_fork_ids;
};

// Keep the template only on the derived class because ``RRefContext`` needs to
// erase the type on ``RRef`` and keep them in one map.
template <typename T>
class RRefImpl final: public RRef {
 public:
  RRefImpl(worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId)
      : RRef(ownerId, rrefId, forkId) {}

  RRefImpl(RRefImpl<T>&& other) noexcept
      : RRef(other.owner(), other.id(), other.forkId()),
        value_(other.value_) {}

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

  IValue getValue() override {
    if(std::is_same<T, IValue>::value) {
      // TODO: use callback to make this non-blocking
      std::unique_lock<std::mutex> lock(mutex_);
      valueCV_.wait(lock, [this]{return value_.has_value();});
      return value_.value();
    } else {
      AT_ERROR("Trying to store an IValue in incompatible RRef[T].");
    }
  }

 private:
  c10::optional<T> value_;
  std::mutex mutex_;
  std::condition_variable valueCV_;
};


} // namespace rpc
} // namespace distributed
} // namespace torch
