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

  RRefContext(const RRefContext &) = delete;
  void operator=(const RRefContext &) = delete;

  worker_id_t getWorkerId() const;
  RRefId genRRefId();
  const std::shared_ptr<RpcAgent>& agent() const;

  // create a new RRef
  template <typename T>
  std::shared_ptr<OwnerRRef<T>> createOwnerRRef(worker_id_t ownerId) {
    TORCH_CHECK(ownerId == getWorkerId(), "Cannot create OwnerRRef on user.");
    return getOrCreateOwnerRRef<T>(genRRefId());
  }

  template <typename T>
  std::shared_ptr<UserRRef<T>> createUserRRef(worker_id_t ownerId) {
    TORCH_CHECK(ownerId != getWorkerId(), "Cannot create UserRRef on owner.");
    return createUserRRef<T>(ownerId, genRRefId(), genRRefId());
  }

  template <typename T>
  std::shared_ptr<UserRRef<T>> createUserRRef(
      worker_id_t ownerId, RRefId rrefId, ForkId forkId) {
    TORCH_CHECK(ownerId != getWorkerId(), "RRef owner cannot create user RRef.");
    // RRefContext does not track user RRefs, it will be destructed when there is
    // no shared_ptrs pointing to it.
    // NB: cannot use make_shared here as the constructor of UserRRef is private
    auto userRRef =
        std::shared_ptr<UserRRef<T>>(new UserRRef<T>(ownerId, rrefId, forkId));

    {
      std::lock_guard<std::mutex> lock(mutex_);
      TORCH_CHECK(pendingUsers_.find(forkId) == pendingUsers_.end(),
          "Inconsistent state, attempt to create the same UserRRef twice.")

      auto iter = pendingAcceptedUsers_.find(forkId);
      if (iter == pendingAcceptedUsers_.end()) {
        // UserRRef created before receiving RREF_USER_ACCEPT message
        pendingUsers_[forkId] = userRRef;
      } else {
        // RREF_USER_ACCEPT arrives before UserRRef is created, remove it
        pendingAcceptedUsers_.erase(iter);
      }
    }
    return userRRef;
  }

  // get an existing RRef or create a new one from a serialized
  // ``RRefForkData``.
  template <typename T>
  std::shared_ptr<RRef> getOrCreateRRef(at::IValue&& value) {
    auto rfd = RRefForkData::fromIValue(std::move(value));
    return getOrCreateRRef<T>(rfd.ownerId_, rfd.rrefId_, rfd.forkId_);
  }

  template <typename T>
  std::shared_ptr<RRef> getOrCreateRRef(
      worker_id_t ownerId, RRefId rrefId, ForkId forkId) {
    if (ownerId == getWorkerId()) {
      return getOrCreateOwnerRRef<T>(rrefId);
    } else {
      return createUserRRef<T>(ownerId, rrefId, forkId);
    }
  }

  template <typename T>
  std::shared_ptr<OwnerRRef<T>> getOrCreateOwnerRRef(RRefId rrefId) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = owners_.find(rrefId);
    if (iter == owners_.end()) {
      // Scenario (1) the first time this owner knows about this RRef
      // Scenario (2) This owner is also the creator.
      //
      // NB: cannot use make_shared here as the constructor of OwnerRRef is
      // private.
      auto rref = std::shared_ptr<OwnerRRef<T>>(
          new OwnerRRef<T>(getWorkerId(), rrefId));
      owners_[rref->id()] = rref;
      return rref;

    } else {
      // Scenario (3) retrieving an existing RRef
      return std::dynamic_pointer_cast<OwnerRRef<T>>(iter->second);
    }
  }

  void acceptUserRRef(
      const RRefId& rrefId, const ForkId& forkId, worker_id_t user);

  RRefForkData forkTo(std::shared_ptr<RRef>, worker_id_t forkDst);
  void acceptForkRequest(IValue request, worker_id_t forkDst);
  void finishForkRequest(IValue request);
  void finishUserRRef(IValue forkId);

  void addForkOfOwner(at::IValue&& value);
  void addForkOfOwner(const RRefId& rrefId, const ForkId& forkId);
  void delForkOfOwner(at::IValue&& value);
  void delForkOfOwner(const RRefId& rrefId, const ForkId& forkId);

 private:
  RRefContext(std::shared_ptr<RpcAgent>);

  static std::unique_ptr<RRefContext> context_;
  static std::atomic<local_id_t> nextLocalId_;

  const std::shared_ptr<RpcAgent> agent_;
  std::mutex mutex_;
  // Keep OwnerRRefs alive until there is no living UserRRefs.
  std::unordered_map<RRefId, std::shared_ptr<RRef>, RRefId::Hash> owners_;
  // Tracks known living UserRRefs of an OwnerRRef
  std::unordered_map<RRefId,
                     std::unordered_set<ForkId, ForkId::Hash>,
                     RRefId::Hash> forks_;

  // The follow two maps keep UserRRefs alive by holding a shared_ptr to the
  // RRef instances. A UserRRef must be added into this map if any of the
  // following two conditions is ture:
  //
  // (1) A UserRRef has not been accepted by owner yet.
  //
  //     It can be used or shared, but cannot be deleted, and hence in this map.
  //     A message of type RREF_USER_ACCEPT will remove the corresponding RRef
  //     from this map.
  std::unordered_map<ForkId,
                     std::shared_ptr<RRef>,
                     ForkId::Hash> pendingUsers_;

  // (2) A UserRRef has pending fork requests that are not accepted by the owner
  //     yet.
  //
  //     This is case, this UserRRef cannot send out RREF_USER_DELETE message,
  //     because it is not guaranteed communications are FIFO between any pair
  //     of worker (due to thread pool and potentially new RpcAgent
  //     implementations). As a result, RREF_USER_DELETE might be processed
  //     by the owner before previous RREF_FORK_NOTIFY messages, which would
  //     mess up RRef reference counts.
  std::unordered_map<ForkId,
                     std::shared_ptr<RRef>,
                     ForkId::Hash> pendingForkRequests_;

  // RREF_USER_ACCEPT message arrives before the UserRRef was created. This may
  // occur as the RREF_USER_ACCEPT is sent from owner to the callee UserRRef,
  // while the UserRRef is created when the message from caller UserRRef arrives
  std::unordered_set<ForkId, ForkId::Hash> pendingAcceptedUsers_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
