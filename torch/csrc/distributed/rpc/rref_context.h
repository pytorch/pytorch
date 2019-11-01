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

  RRefContext(const RRefContext&) = delete;
  void operator=(const RRefContext&) = delete;

  worker_id_t getWorkerId() const;
  RRefId genRRefId();
  const std::shared_ptr<RpcAgent>& agent() const;

  // create a new RRef
  template <typename T>
  std::shared_ptr<OwnerRRef<T>> createOwnerRRef(worker_id_t ownerId) {
    TORCH_CHECK(ownerId == getWorkerId(), "Cannot create OwnerRRef on user.");
    return getOrCreateOwnerRRef<T>(genRRefId());
  }

  std::shared_ptr<UserRRef> createUserRRef(worker_id_t ownerId) {
    TORCH_CHECK(ownerId != getWorkerId(), "Cannot create UserRRef on owner.");
    return createUserRRef(ownerId, genRRefId(), genRRefId());
  }

  std::shared_ptr<UserRRef> createUserRRef(
      worker_id_t ownerId,
      const RRefId& rrefId,
      const ForkId& forkId) {
    TORCH_CHECK(
        ownerId != getWorkerId(), "RRef owner cannot create user RRef.");
    // RRefContext does not track user RRefs, it will be destructed when there
    // is no shared_ptrs pointing to it. NB: cannot use make_shared here as the
    // constructor of UserRRef is private
    return std::shared_ptr<UserRRef>(new UserRRef(ownerId, rrefId, forkId));
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
      worker_id_t ownerId,
      const RRefId& rrefId,
      const ForkId& forkId) {
    if (ownerId == getWorkerId()) {
      return getOrCreateOwnerRRef<T>(rrefId);
    } else {
      return createUserRRef(ownerId, rrefId, forkId);
    }
  }

  template <typename T>
  std::shared_ptr<OwnerRRef<T>> getOrCreateOwnerRRef(const RRefId& rrefId) {
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

  void addFork(const at::IValue& value);
  void delFork(const at::IValue& value);

 private:
  RRefContext(std::shared_ptr<RpcAgent>);

  static std::unique_ptr<RRefContext> context_;
  static std::atomic<local_id_t> nextLocalId_;

  const std::shared_ptr<RpcAgent> agent_;
  std::mutex mutex_;
  // Keep OwnerRRefs alive until there is no living UserRRefs.
  std::unordered_map<RRefId, std::shared_ptr<RRef>, RRefId::Hash> owners_;
  // Tracks known living UserRRefs of an OwnerRRef
  std::unordered_map<
      RRefId,
      std::unordered_set<ForkId, ForkId::Hash>,
      RRefId::Hash>
      forks_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
