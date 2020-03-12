#pragma once

#include <ATen/core/jit_type.h>
#include <ATen/core/rref_interface.h>
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

constexpr int OWNER_IDX = 0; // index of ownerId in the tuple
constexpr int RREFID_ON_IDX = 1; // index of RRefId.createdOn_ in the tuple
constexpr int RREFID_ID_IDX = 2; // index of RRefId.localId_ in the tuple
constexpr int FORKID_ON_IDX = 3; // index of ForkId.createdOn_ in the tuple
constexpr int FORKID_ID_IDX = 4; // index of ForkId.localId_ in the tuple
constexpr int PARENT_IDX = 5; // index of parent in the tuple
constexpr int TYPE_IDX = 6; // index of parent in the tuple

// NB: if more fields are added, make sure this field is also bumped
constexpr int RFD_TUPLE_SIZE = 7; // number of RRefForkData fields in py::tuple

// Represents fork of an RRef to be sent over the wire.
struct TORCH_API RRefForkData {
  const worker_id_t ownerId_;
  const RRefId rrefId_;
  const ForkId forkId_;
  const worker_id_t parent_;
  const std::string typeStr_;

  RRefForkData(
      worker_id_t ownerId,
      const RRefId& rrefId,
      const ForkId& forkId,
      worker_id_t parent,
      std::string typeStr);
};

// Note [RRef Protocol]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// [Background]
//
// RRef stands for Remote REFerence. Each RRef is owned by a single worker
// (i.e., owner) and can be used by multiple users. The owner stores the real
// data referenced by its RRefs. RRef needs to support fast and scalable RPC.
// Hence, in the design, we avoid using a single global master to keep RRef
// states, instead owners will keep track of the global reference counts
// for its RRefs. Every RRef can be uniquely identified by a global RRefId,
// which is assigned at the time it is first created either on a user or on the
// owner.
//
// On the owner worker, there is only one OwnerRRef instance, which contains the
// real data, while on user workers, there can be as many UserRRefs as
// necessary, and UserRRef does not hold the data. All usage on the OwnerRRef
// should retrieve the unique OwnerRRef instance using the globally unique
// RRefId. //A UserRRef will be created when it is used as an argument or return
// value in dist.rpc or dist.remote call, but RRef forking and reference
// counting (RC) are completely transparent to applications. Every UserRRef will
// also have its globally unique ForkId.
//
// [Assumptions]
//
// 1. Transient Network Failures
//
// TODO: current RRef implementation does not tolerate failures
//
// The RRef design aims to handle transient network failures by retrying
// messages. Node crashes or permanent network partition is beyond the scope.
// When those incidents occur, the application may take down all workers, revert
// to the previous checkpoint, and resume training.
//
// 2. Non-idempotent UDFs
//
// We assume UDFs are not idempotent and therefore cannot be retried. However,
// internal RRef control messages will be made idempotent and retryable.
//
// TODO: RRef internal messages are not yet idempotent
//
// 3. Out of Order Message Delivery
//
// We do not assume message delivery order between any pair of nodes, because
// both sender and receiver are using multiple threads. There is no guarantee on
// which message will be processed first.
//
// [RRef Lifetime]
//
// The goal of the protocol is to delete an OwnerRRef at an appropriate time.
// The right time to delete an OwnerRRef is when there are no living UserRRefs
// and Python GC also agrees to delete the OwnerRRef instance on the owner. The
// tricky part is to determine if there are any living UserRRefs.
//
// A user can get a UserRRef in three situations:
//
// (1). Receiving a UserRRef from the owner.
// (2). Receiving a UserRRef from another user.
// (3). Creating a new UserRRef owned by another worker.
//
// (1) is the simplest case where the owner initiates the fork, and hence it can
// easily increment local RC. The only requirement is that any UserRRef must
// notify the owner before destruction. Hence, we need the first guarantee:
//
// G1. The owner will be notified when any UserRRef is deleted.
//
// As messages might come delayed or out-of-order, we need more one guarantee to
// make sure the delete message is not sent out too soon. Let us first introduce
// a new concept. If A sends an RPC to B that involves an RRef, we call the RRef
// on A the parent RRef and the RRef on B the child RRef.
//
// G2. Parent RRef cannot be deleted until the child RRef is confirmed by the
//     owner.
//
// Under (1), where the caller is UserRRef and callee is OwnerRRef, it simply
// means that the user will not send out the delete message until all previous
// messages are ACKed. Note that ACKed does not mean the owner finishes
// executing the function, instead, it only means the owner has retrieved its
// local OwnerRRef and about to pass it to the function, which is sufficient to
// keep the OwnerRRef alive even if the delete message from the user arrives at
// the owner before the function finishes execution.
//
// With (2) and (3), it is possible that the owner only partially knows the RRef
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
//
//       1) unknown -> 2) known -> 3) deleted.
//
// The owner's view on the entire tree keeps changing. The owner deletes its
// OwnerRRef instance when it thinks there are no living UserRRefs, i.e., when
// OwnerRRef is deleted, all UserRRefs could be either indeed deleted or
// unknown. The dangerous case is when some forks are unknown and others are
// deleted.
//
// G2 trivially guarantees that no parent UserRRef Y can be deleted before the
// owner knows all of Y's children UserRRefs.
//
// However, it is possible that the child UserRRef Z may be deleted before the
// owner knows its parent Y. More specifically, this can happen when all of Z's
// messages are processed by the owner before all messages from Y, including the
// delete message. Nevertheless, this does not cause any problem. Because, at
// least one of Y's ancestor will be alive, and it will prevent the owner from
// deleting the OwnerRRef. Consider the following example: (NB: this scenario
// will no longer relevant when we block UDF until all RRefs are confirmed by
// the owner)
//
//     OwnerRRef -> A -> Y -> Z
//
// OwnerRRef forks to A, then A forks to Y, and Y forks to Z. Z can be deleted
// without OwnerRRef knowing Y. However, the OwnerRRef will at least know A, as
// the owner directly forks the RRef to A. A won't die before the owner knows Y.
//
// Things get a little trickier if the RRef is created on a user:
//
//  OwnerRRef
//      ^
//      |
//      A -> Y -> Z
//
// If Z calls to_here on the UserRRef, the owner at least knows A when Z is
// deleted, because otherwise to_here wouldn't finish. If Z does not call
// to_here, it is possible that the owner receives all messages from Z before
// any message from A and Y. In this case, as the real data of the OwnerRRef has
// not been created yet, there is nothing to be deleted either. It is the same
// as Z does not exist at all Hence, it's still OK.
//
// See #26759 for more details and discussions.
//
// TODO: make RRef an IValue, and edit createStackForSchema accordingly
// TODO: make RRef system messages idempotent and retry on failures.
//
// ``RRef`` is the base type for both ``UserRRef`` and ``OwnerRRef``.
// Each ``RRef`` has a globally unique ``RRefId``.
class TORCH_API RRef : public RRefInterface {
 public:
  // RRef is made NOT copyable NOT movable to prevent messing up reference
  // counting.
  explicit RRef(const RRef& other) = delete;
  explicit RRef(RRef&& other) = delete;
  RRef& operator=(RRef&& other) = delete;

  virtual ~RRef() = default;

  // returns the worker id of the owner
  inline worker_id_t owner() const override {
    return ownerId_;
  }

  // Returns the globally unique RRefId of this RRef
  inline const RRefId& rrefId() const {
    return rrefId_;
  }

  inline bool isPyObj() {
    return type_ == PyObjectType::get();
  }
  inline const TypePtr type() const override {
    return type_;
  }

  // Send delete UserRRef request to Owner,
  // if the request hasn't been sent yet.
  // There are 2 cases to call it,
  // 1, Python GC decides end of UserRRef lifetime, calling destructor.
  // 2, RPC module graceful shutdown calls it on all UserRRefs tracked
  //    in the RRefContext.
  virtual void tryDel() {}

 protected:
  friend class RRefContext;

  RRef(worker_id_t ownerId, const RRefId& rrefId, TypePtr type);

  RRefForkData fork() const;

  const worker_id_t ownerId_;
  const RRefId rrefId_;

  // type field to denote the type of the element that the RRef is holding
  // it could be any TypePtr that JIT support, including PyObjectType
  const TypePtr type_;
};

// ``UserRRef`` represents a user of an RRef. Besides the ``RRefId``, each user
// also has a globally unique ``ForkId`` to identify this user. ``UserRRef``
// never owns the real value, the only way to get the value of the ``RRef`` is
// to call ``to_here()`` and get a copy..
class TORCH_API UserRRef final : public RRef {
 public:
  UserRRef(const UserRRef& other) = delete;
  UserRRef(UserRRef&& other) = delete;
  UserRRef& operator=(const UserRRef& other) = delete;
  UserRRef& operator=(UserRRef&& other) = delete;

  UserRRef(
      worker_id_t ownerId,
      const RRefId& rrefId,
      const ForkId& forkId,
      TypePtr type);

  inline bool isOwner() const override {
    return false;
  }

  inline bool isConfirmed() const override {
    return confirmed_;
  }

  // Returns the globally unique ForkId of this RRef
  const ForkId& forkId() const;

  // Get of copy of the value from the ``OwnerRRef``. If the value is not ready
  // yet, this call will block.
  IValue toHere();

  void tryDel() override;

  // Will be called when refcount reaches 0.
  // Upon destruction, this ``UserRRef`` will tell the owner to deref.
  void release_resources() override;

  // Will be called when both refcount and weakcount reach 0. See
  // https://github.com/pytorch/pytorch/blob/9116f02bebf3a5260feef5732d36c54ecb3b4033/c10/util/intrusive_ptr.h#L204
  // This is called on destructing the wrapping intrusive_ptr_target instance
  // and it's data members. We don't need to implement anything here.
  ~UserRRef() = default;

 private:
  friend class RRefContext;

  inline void confirm() {
    confirmed_ = true;
  }

  const ForkId forkId_;

  // Indicates if this user has sent delete message to it's owner.
  // Note, thread safety is needed because delete message could be sent by
  // either the destructor called by Python garbage collection or RRefContext
  // proactive cleanup on RPC graceful shutdown.
  std::mutex deletedOnOwnerMutex_;
  bool deletedOnOwner_{false};
  std::atomic<bool> confirmed_;
};

// Keep the template only on the derived class because ``RRefContext`` needs to
// erase the type on ``RRef`` and keep them in one map.
class TORCH_API OwnerRRef final : public RRef {
 public:
  OwnerRRef(const OwnerRRef& other) = delete;
  OwnerRRef(OwnerRRef&& other) = delete;
  OwnerRRef& operator=(const OwnerRRef& other) = delete;
  OwnerRRef& operator=(OwnerRRef&& other) = delete;

  OwnerRRef(worker_id_t ownerId, const RRefId& rrefId, TypePtr type)
      : OwnerRRef(ownerId, rrefId, type, {}) {}

  OwnerRRef(
      worker_id_t ownerId,
      const RRefId& rrefId,
      TypePtr type,
      c10::optional<IValue> value)
      : RRef(ownerId, rrefId, std::move(type)) {
    value_ = std::move(value);
  }

  inline bool isOwner() const override {
    return true;
  }

  // OwnerRRef is always confirmed, while UserRRef is only confirmed when the
  // owner knows about it.
  inline bool isConfirmed() const override {
    return true;
  }

  // Get a constant reference of the real value. This method will block if the
  // value is not ready. This method does not need GIL as it does not create
  // any new py::object.
  const IValue& getValue() const;

  // Set the value of this ``OwnerRRef``. This method does not need GIL as it
  // does not create any new py::object.
  void setValue(IValue&& value);

  // Has a value been set?
  bool hasValue() const;
  // Gets a future that is satisfied when the value is set.
  std::shared_ptr<FutureMessage> getFuture();

 private:
  friend class RRefContext;

  c10::optional<IValue> value_;
  mutable std::mutex mutex_;
  mutable std::condition_variable valueCV_;
  std::shared_ptr<FutureMessage> future_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
