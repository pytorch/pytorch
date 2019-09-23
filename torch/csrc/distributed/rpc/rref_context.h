#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/distributed/rpc/types.h>

#include <atomic>

namespace torch {
namespace distributed {
namespace rpc {

// Manages RRef lifetime and keeps track of RRef forks.
class RRefContext {
 public:
  static void initInstance(std::shared_ptr<RpcAgent>);
  static std::unique_ptr<RRefContext>& getInstance();

  static void handleException(const Message& message);

  RRefContext(const RRefContext&) = delete;
  RRefContext(RRefContext&& other) = delete;
  void operator=(const RRefContext&) = delete;
  RRefContext& operator=(RRefContext&& other) = delete;

  ~RRefContext();

  // get the worker id of the current worker
  inline worker_id_t getWorkerId() const {
    return agent_->getWorkerInfo().id_;
  }

  // get the worker name of the current worker
  inline const std::string& getWorkerName() const {
    return agent_->getWorkerInfo().name_;
  }

  //  generate a globally unique ID
  inline GloballyUniqueId genGloballyUniqueId() {
    return GloballyUniqueId(getWorkerId(), nextLocalId_++);
  }

  inline const std::shared_ptr<RpcAgent>& agent() const {
    return agent_;
  }

  // create a ``UserRRef`` owned by the worker ``ownerId``
  template <typename T>
  std::shared_ptr<UserRRef<T>> createUserRRef(worker_id_t ownerId);

  // Convert an RRefForkData into an RRef. This RRef could be user or owner.
  // This RRef could have already existed before, or could be created in this
  // method.
  template <typename T>
  std::shared_ptr<RRef> getOrCreateRRef(const RRefForkData& rfd);

  // Get the ``OwnerRRef`` of id ``rrefId``. If it does not exist, create a new
  // one.
  template <typename T>
  std::shared_ptr<OwnerRRef<T>> getOrCreateOwnerRRef(const RRefId& rrefId);

  // Register a fork of the ``OwnerRRef``, and inserts a shared_ptr of the
  // ``OwnerRRef`` in a map to keep it alive.
  void addForkOfOwner(const RRefId& rrefId, const ForkId& forkId);
  // Delete a fork of the ``OwnerRRef``. NB: this could trigger deletion on the
  // IValue or py::object. For the later, this method will acquire GIL.
  void delForkOfOwner(const RRefId& rrefId, const ForkId& forkId);

  // Invoked when pickling an RRef to setup child/fork properly
  RRefForkData prepareChildFork(const std::shared_ptr<RRef>& rref);
  // Invoked when unpickling an RRef to send RREF_FORK_REQUEST to owner and
  // send RREF_CHILD_ACCEPT to the parent.
  // NB: forkId is necessary here as the rref could be an OwnerRRef
  void notifyOwnerAndParentOfFork(
      const ForkId& forkId,
      worker_id_t parent,
      const std::shared_ptr<RRef>& rref);

  // When a UserRRef is forked to another worker (user or owner), it is added
  // into pendingChildren_ to be held alive until it receives RREF_CHILD_ACCEPT
  // from the child.
  // NB: This is necessary for both user and owner child. As we do not have FIFO
  // communication between workers, we need this strategy to make sure that all
  // previously submitted rpc/remote calls are acked before sending out the
  // RREF_USER_DELETE message. Otherwise, the OwnerRRef could be deleted too
  // soon.
  template <typename T>
  void addPendingChild(
      const ForkId& forkId,
      const std::shared_ptr<UserRRef<T>>& rref);
  void delPendingChild(const ForkId& forkId);

  // When a UserRRef is created, it is added into pendingUsers_ to be held alive
  // until it receives RREF_USER_ACCEPT from the owner.
  template <typename T>
  void addPendingUser(
      const ForkId& forkId,
      const std::shared_ptr<UserRRef<T>>& rref);
  void delPendingUser(const ForkId& forkId);

  // If there is any leak on any RRef, this method will throw an error.
  void checkRRefLeaks();

 private:
  RRefContext(std::shared_ptr<RpcAgent>);

  template <typename T>
  std::shared_ptr<UserRRef<T>> createUserRRef(
      worker_id_t ownerId,
      const RRefId& rrefId,
      const ForkId& forkId);

  void finishForkRequest(const ForkId& forkId, worker_id_t parent);

  static std::unique_ptr<RRefContext> context_;
  static std::atomic<local_id_t> nextLocalId_;

  const std::shared_ptr<RpcAgent> agent_;
  mutable std::mutex mutex_;
  // Keep OwnerRRefs alive until there is no living UserRRefs.
  std::unordered_map<RRefId, std::shared_ptr<RRef>, RRefId::Hash> owners_;
  // Tracks known living UserRRefs of an OwnerRRef
  std::unordered_map<
      RRefId,
      std::unordered_set<ForkId, ForkId::Hash>,
      RRefId::Hash>
      forks_;

  // The follow two maps keep UserRRefs alive by holding a shared_ptr to the
  // RRef instances. A UserRRef must be added into this map if any of the
  // following two conditions is ture:
  //
  // (1) A UserRRef has not been accepted by owner yet.
  //
  //     It can be used or shared, but cannot be deleted, and hence kept alive
  //     in this map. A message of type RREF_USER_ACCEPT will remove the
  //     corresponding RRef from this map.
  std::unordered_map<ForkId, std::shared_ptr<RRef>, ForkId::Hash> pendingUsers_;

  // (2) A UserRRef has forked a child UserRRef which has not been accepted by
  //     the owner yet.
  //
  //     In this case, this UserRRef cannot send out RREF_USER_DELETE message,
  //     as it could potentially trigger the OwnerRRef been deleted before the
  //     owner learns about the forked child.
  std::unordered_map<ForkId, std::shared_ptr<RRef>, ForkId::Hash>
      pendingChildren_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
