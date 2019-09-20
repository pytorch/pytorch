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
struct RRefForkData {
  at::IValue toIValue() const;

  py::tuple toPyTuple() const;
  static RRefForkData fromPyTuple(const py::tuple& obj);

  const worker_id_t ownerId_;
  const RRefId rrefId_;
  const ForkId forkId_;
  const worker_id_t parent_;

 private:
  friend class RRef;
  friend class RRefContext;
  template <typename T>
  friend class UserRRef;

  RRefForkData(
      worker_id_t ownerId,
      const RRefId& rrefId_,
      const ForkId& forkId_,
      worker_id_t parent);

  static RRefForkData fromIValue(const at::IValue&);
};

static_assert(
    C10_IS_TRIVIALLY_COPYABLE(RRefForkData),
    "RRefForkData must be trivially copyable");

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
//    instance. At the same time, the caller holds its own RRef alive until it
//    receives RREF_CHILD_ACCEPT from the owner.
// 3. If the RPC is between two users:
//    a. The caller serializes the RRef into a RRefForkData and includes it in
//       the RPC message sent to the callee. At the time, the caller holds
//       its own RRef alive until it receives RREF_CHILD_ACCEPT from the callee.
//    b. Upon receiving the RPC message, the callee sends an RREF_FORK_REQUEST
//       to the owner. When the RREF_USER_ACCEPT from owner arrives, the callee
//       runs the UDF and sends a RREF_CHILD_ACCEPT message to the caller.
//       TODO: Currently, the callee runs the UDF immediately after getting the
//       RPC message instead of waiting for all UserRRefs being confirmed by the
//       owner.
//    b. The owner, upon receiving the notification, updates its local RC.
//
// NB: RREF_FORK_REQUEST only registers the callee UserRRef on the owner, not
// the caller. So, it is possible that the owner knows the callee UserRRef
// before knowing the caller UserRRef, and it is possible that the callee
// UserRRef even gets deleted before the owner knows the caller RRef.
//
//
// Note [RRef Reference Count]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// The right time to delete an OwnerRRef is when there are no living UserRRefs
// and Python GC also agrees to delete the OwnerRRef instance on the
// owner. The tricky part is to determine if there are any living UserRRefs.
//
// A user can get a UserRRef in three situations:
//
// 1. Receiving a UserRRef from the owner.
// 2. Receiving a UserRRef from another user.
// 3. Creating a new UserRRef owned by another worker.
//
// #1 is the simplest case where the owner initiates the fork, and hence it can
// easily increment local RC. The only requirement is that any UserRRef must
// notify the owner before destruction. Hence, we need the first guarantee:
//
// G1. The owner will be notified when any UserRRef is deleted.*
//
// Note that the notification might come delayed or out-of-order.
//
// With #2 and #3, it is possible that the owner only partially knows the RRef
// fork graph or not even knowing it at all. For example, the RRef could be
// constructed on a user, and before the owner receives the RPC call, the
// creator user might have already shared the RRef with other users, and those
// users could further share the RRef. One invariant is that the fork graph of
// any RRef is always a tree rooted at the owner, because forking an RRef always
// creates a new RRef instance, and hence every RRef has a single parent. One
// nasty detail is that when an RRef is created on a user, technically the owner
// is not its parent but we still consider it that way and it does not break the
// argument below.
//
// The owner's view on any node (fork) in the tree has three stages:
//            1) unknown → 2) known → 3) deleted.
// The owner's view on the entire tree keeps changing. The owner deletes its
// OwnerRRef instance when it thinks there are no living UserRRefs, i.e., all
// the UserRRefs could be either indeed deleted or unknown. The dangerous case
// is when some forks are unknown and others are deleted. We only need a simple
// guarantee to prevent this situation:
//
// * G2. Parent UserRRef cannot be deleted until the child UserRRef is confrimed
//       by the owner.
//
// G2 trivially guarantees that no parent UserRRef Y can be deleted before the
// owner knows all Y's children UserRRefs.
//
// However, it is possible that a child UserRRef Z is deleted before the owner
// knows its Z's parent Y. More specifically, this can happen when Z's
// RREF_FORK_REQUEST was processed by the owner before any other messages from
// Y, where Z receives RREF_USER_ACCEPT from the owner and then send out
// RREF_USER_DELETE before the owner leanrs about Y. Nevertheless, this does not
// cause any problem. Because, at least one of Y's ancestor will be alive if Z,
// preventing the owner from deleting the OwnerRRef. Consider the following
// example:
//
//    OwnerRRef -> A -> Y -> Z
//
// OwnerRRef forks to A, then A forks to B, B forks to Y, and Y forks to Z. Z
// can be deleted without OwnerRRef knowing Y or even B. However, the OwnerRRef
// will at least know A, A won't die before the owner knows Y.
//
// Things get a little trickier if the RRef starts from a user:
//
// OwnerRRef
//     ^
//     |
//     A -> Y -> Z
//
// If Z calls to_here on the UserRRef, the owner at least knows A when Z is
// deleted, because otherwise to_here wouldn't finish. If Z does not call
// to_here, it is possible that the owner receives all messages from Z before
// any message from A and Y. In this case, as the OwnerRRef hasn't been created
// on the owner yet (it is only created when owner receives a remote call or a
// to_here() call), there is nothing to be deleted either. Hence, it's still OK.
//
//
// TODO: make RRef an IValue, and edit createStackForSchema accordingly
// TODO: make RRef system messages idempotent and retry on failures.
class RRef {
 public:
  // RRef is made NOT copyable NOT movable to prevent messing up reference
  // counting
  RRef(const RRef& other) = delete;
  RRef(RRef&& other) = delete;

  virtual ~RRef() = default;

  inline worker_id_t owner() const {
    return ownerId_;
  }

  inline const RRefId& rrefId() const {
    return rrefId_;
  }

  virtual bool isOwner() const = 0;

  // returns true if this RRef holds an py::object, false if IValue
  virtual bool isPyObj() = 0;

 protected:
  friend class RRefContext;

  RRef(worker_id_t ownerId, const RRefId& rrefId);

  RRefForkData fork() const;

  const worker_id_t ownerId_;
  const RRefId rrefId_;
};

template <typename T>
class UserRRef final : public RRef {
 public:
  UserRRef(const UserRRef& other) = delete;
  UserRRef(UserRRef&& other) = delete;
  UserRRef& operator=(const UserRRef other) = delete;
  UserRRef& operator=(UserRRef&& other) = delete;

  inline bool isOwner() const override {
    return false;
  }

  inline bool isPyObj() override {
    return std::is_same<T, py::object>::value;
  }

  const ForkId& forkId() const;
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
class OwnerRRef final : public RRef {
 public:
  OwnerRRef(const OwnerRRef& other) = delete;
  OwnerRRef(OwnerRRef&& other) = delete;
  OwnerRRef& operator=(const OwnerRRef other) = delete;
  OwnerRRef& operator=(OwnerRRef&& other) = delete;

  inline bool isOwner() const override {
    return true;
  }

  inline bool isPyObj() override {
    return std::is_same<T, py::object>::value;
  }

  const T& getValue() const;
  void setValue(T&& value);

 private:
  friend class RRefContext;

  OwnerRRef(worker_id_t ownerId, const RRefId& rrefId)
      : OwnerRRef(ownerId, rrefId, {}) {}

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
