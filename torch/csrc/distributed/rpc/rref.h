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

  RRefForkData(
      worker_id_t ownerId,
      const RRefId& rrefId_,
      const ForkId& forkId_);

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
// NB: RREF_FORK_NOTIFY only registers the callee UserRRef on the owner, not the
// caller. So, it is possible that the owner knows the callee UserRRef before
// knowing the caller UserRRef. This design decision is made to simplify the
// protocole. If RREF_FORK_NOTIFY registers both UserRRefs, RREF_USER_ACCEPT
// might be sent to the caller UserRRef under two different situations, and will
// lead to more states tracking and dedup complexities. As we will see below in
// [RRef Reference Count], reference count can still work properly with this
// simplification.
//
//
// Note [RRef Reference Count]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// The right time to delete an RRef on owner is when there are no living forks
// on any user and Python GC also agrees to delete the RRef instance on the
// owner. The tricky part is to determine if there are any living forks.
//
// A user can get a fork in three situations:
//
// 1. Receiving a fork from the owner.
// 2. Receiving a fork from another user.
// 3. Creating a new RRef fork owned by another worker.
//
// #1 is the simplest case where the owner initiates the fork, and hence it can
// easily increment local RC. The only requirement is that any fork must notify
// the owner before destruction. Hence, we need the first guarantee:
//
// G1. The owner will be notified when any fork is deleted.*
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
// RRef instance when it thinks there are no living forks, i.e., all the forks
// could be either indeed deleted or unknown. Therefore, the dangerous case is
// when some forks are unknown and others are deleted. We only need a simple
// guarantee to prevent this situation:
//
// * G2. No fork x can be deleted if it has any unacknowledged fork requests.
//
// G2 trivially guarantees that no parent UserRRef Y can be deleted before the
// owner knows all Y's children UserRRefs.
//
// However, it is possible that a child UserRRef Z is deleted before the owner
// knows its Z's parent Y. More specifically, this can happen when Y's
// RREF_FORK_NOTIFY was processed by the owner before any other messages from Y,
// where Z can RREF_USER_ACCEPT and then send out RREF_USER_DELETE before the
// owner leanrs about Y. Nevertheless, this does not cause any problem. Because,
// at least one of Y's ancestor will be alive, preventing the owner from
// deleting the OwnerRRef. Consider the following example:
//
//    OwnerRRef -> A -> B -> Y -> Z
//
// OwnerRRef forks to A, then A forks to B, B forks to Y, and Y forks to Z. Z
// can be deleted without OwnerRRef knowing Y or even B. However, the OwnerRRef
// will at least know A, and A won't die before the owner knows B, and B won't
// die before the owner knows Y.
//
// In general, on any root-to-leaf path in the RRef sharing graph, when any
// UserRRef leaves, the owner will always know its child. Therefore, for any
// path formed by unknown and known RRefs, the first node on the path will
// always be known. Hence, the OwnerRRef will not be deleted as long as there is
// any living UserRRef.
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
  bool isOwner() const override;
  bool isPyObj() override;

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
  bool isOwner() const override;
  bool isPyObj() override;

  T getValue() const;
  void setValue(T&& value);

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
