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
  void operator=(const RRefContext&) = delete;

  ~RRefContext();

  inline worker_id_t getWorkerId() const {
    return agent_->getWorkerInfo().id_;
  }

  inline const std::string& getWorkerName() const {
    return agent_->getWorkerInfo().name_;
  }

  inline GloballyUniqueId genGloballyUniqueId() {
    return GloballyUniqueId(getWorkerId(), nextLocalId_++);
  }

  inline const std::shared_ptr<RpcAgent>& agent() const {
    return agent_;
  }

  template <typename T>
  std::shared_ptr<UserRRef<T>> createUserRRef(worker_id_t ownerId);

  template <typename T>
  std::shared_ptr<RRef> getOrCreateRRef(const RRefForkData& rfd);

  template <typename T>
  std::shared_ptr<OwnerRRef<T>> getOrCreateOwnerRRef(const RRefId& rrefId);

  Message acceptUserRRef(const RRefId& rrefId, const ForkId& forkId);
  void finishForkRequest(const ForkId& forkId, worker_id_t parent);

  void addForkOfOwner(const RRefId& rrefId, const ForkId& forkId);
  std::shared_ptr<RRef> delForkOfOwner(
      const RRefId& rrefId,
      const ForkId& forkId);

  RRefForkData prepareChildFork(const std::shared_ptr<RRef>& rref);
  // forkId is necessary here as the rref could be an OwnerRRef
  void notifyOwnerAndParentOfFork(
      const ForkId& forkId,
      worker_id_t parent,
      const std::shared_ptr<RRef>& rref);

  void addPendingChild(const ForkId& forkId, const std::shared_ptr<RRef>& rref);
  void delPendingChild(const ForkId& forkId);

  void addPendingUser(const ForkId& forkId, const std::shared_ptr<RRef>& rref);
  void delPendingUser(const ForkId& forkId);

 private:
  RRefContext(std::shared_ptr<RpcAgent>);

  template <typename T>
  std::shared_ptr<UserRRef<T>> createUserRRef(
      worker_id_t ownerId,
      const RRefId& rrefId,
      const ForkId& forkId);

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
