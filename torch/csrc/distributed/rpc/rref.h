#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/utils/pybind.h>


#include <atomic>

namespace torch {
namespace distributed {
namespace rpc {


class RRef;
class RRefContext;
template <typename T>
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

  const worker_id_t ownerId_;
  const RRefId rrefId_;
  const ForkId forkId_;
 private:
  friend class RRef;
  friend class RRefContext;
  template <typename T>
  friend class UserRRef;

  RRefForkData(worker_id_t ownerId,
               const RRefId& rrefId_,
               const ForkId& forkId_);

  static RRefForkData fromIValue(at::IValue&&);
};

// Note [RRef Algorithm]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// RRef stands for Remote REFerence. Each RRef is owned by a single worker
// (i.e., owner) and can be used by multiple users. The owner stores the real
// data referenced by its RRefs, and keeps track of the global reference counts
// for its RRefs. Every RRef can be uniquely identified by a global id ref_id,
// which is assigned at the time it is first created either on a user or on the
// owner.
//
// The owner only keeps one RRef instance for each data object, while users can
// fork as many RRef instances as necessary. All usage on the owner should
// retrieve the unique RRef instance using the globally unique ``rrefId``. A
// fork of RRef will be created when it is used as an argument in RPC or the
// return value in a ``remote`` call, but users don't need to worry about
// forking/forwarding and reference counting (RC) RRefs. These will be handled
// transparently. Every fork will also have its own ``forkId``, which is
// guaranteed to be globally unique.
//
// RRef needs to support fast and scalable RPC. Hence, in the RC design, we
// avoid using a single global master to keep RRef states. Besides, when worker
// X invokes RPC on worker Y, Y should be able to start immediately after
// receiving the RPC request, without waiting for any third-party owner Z
// (unless Y needs to pull real data from Z), even if neither X nor Y owns the
// RRef. We propose the following algorithm:
//
// 1. If the owner is the RPC caller, the owner will update RC for the RRef
//    accordingly.
// 2. If the owner is the RPC callee, the owner will drop the new fork, and use
//    the unique RRef id in the fork to access its singleton local RRef
//    instance.
// 3. If the RPC is between two users:
//    a. The caller sends an RPC message to the callee, and also notifies the
//       owner on the new fork (RREF_FORK_NOTIFY).
//    b. The owner, upon receiving the notification, updates its local RC and
//       then:
//        (i). tells the caller the fork request was accept (RREF_FORK_ACCEPT)
//       (ii). tells the callee the new fork is now known by the owner
//             (RREF_USER_ACCEPT).
//    c. The callee can starts executing the RPC as soon as it receives the RPC
//       message from the caller, and does not need to wait for the message from
//       the owner (RREF_USER_ACCEPT). However, it cannot delete
//       (RREF_USER_DELETE) its local RRef fork until owner's message arrives.
//
// Note [RRef Reference Count]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//
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

  virtual bool isOwner() const = 0;
  virtual bool isPyObj() = 0;

 protected:
  friend class RRefContext;

  RRef(worker_id_t ownerId, const RRefId& rrefId);

  RRefForkData fork() const;

  const worker_id_t ownerId_;
  const RRefId rrefId_;
};

template <typename T>
class UserRRef final: public RRef {
 public:
  const ForkId& forkId() const;
  bool isOwner() const override;

  bool isPyObj() override {
    return std::is_same<T, py::object>::value;
  }

  T toHere();

  ~UserRRef() override;
 private:
  friend class RRefContext;

  UserRRef(worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId);

  const ForkId forkId_;
};

// Keep the template only on the derived class because ``RRefContext`` needs to
// erase the type on ``RRef`` and keep them in one map.
template <typename T>
class OwnerRRef final: public RRef {
 public:
  bool isOwner() const override {
    return true;
  }

  T getValue() const {
    // TODO: use callback to make this non-blocking
    std::unique_lock<std::mutex> lock(mutex_);
    valueCV_.wait(lock, [this]{return value_.has_value();});
    return value_.value();
  }

  void setValue(T&& value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      value_ = std::move(value);
    }
    valueCV_.notify_all();
  }

  bool isPyObj() override {
    return std::is_same<T, py::object>::value;
  }

  // TODO: add setValue(py::object) and getPyObj() for Python UDF

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
