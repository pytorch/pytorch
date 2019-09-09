#include <torch/csrc/distributed/rpc/rref_context.h>

#include <torch/csrc/distributed/rpc/script_rref_proto.h>

namespace torch {
namespace distributed {
namespace rpc {

std::unique_ptr<RRefContext> RRefContext::context_;

void RRefContext::initInstance(std::shared_ptr<RpcAgent> agent) {
  TORCH_CHECK(!RRefContext::context_, "Can only initialize RRefContext once.");
  TORCH_CHECK(agent, "RRefContext requires a non-null RpcAgent shared_ptr.");

  RRefContext::context_ =
      std::unique_ptr<RRefContext>(new RRefContext(std::move(agent)));
}

std::unique_ptr<RRefContext>& RRefContext::getInstance() {
  TORCH_CHECK(
      RRefContext::context_, "Have to initialize RRefContext before use.");
  return RRefContext::context_;
}

RRefContext::RRefContext(std::shared_ptr<RpcAgent> agent)
    : agent_(std::move(agent)) {}

worker_id_t RRefContext::getWorkerId() const {
  return agent_->getWorkerId().id_;
}

const std::string& RRefContext::getWorkerName() const {
  return agent_->getWorkerId().name_;
}

RRefId RRefContext::genRRefId() {
  return RRefId(getWorkerId(), nextLocalId_++);
}

const std::shared_ptr<RpcAgent>& RRefContext::agent() const {
  return agent_;
}

template <typename T>
std::shared_ptr<OwnerRRef<T>> RRefContext::createOwnerRRef(
    worker_id_t ownerId) {
  TORCH_CHECK(ownerId == getWorkerId(), "Cannot create OwnerRRef on user.");
  return getOrCreateOwnerRRef<T>(genRRefId());
}

template std::shared_ptr<OwnerRRef<IValue>>
    RRefContext::createOwnerRRef<IValue>(worker_id_t ownerId);

template std::shared_ptr<OwnerRRef<py::object>>
    RRefContext::createOwnerRRef<py::object>(worker_id_t ownerId);

template <typename T>
std::shared_ptr<UserRRef<T>> RRefContext::createUserRRef(worker_id_t ownerId) {
  TORCH_CHECK(ownerId != getWorkerId(), "Cannot create UserRRef on owner.");
  return createUserRRef<T>(ownerId, genRRefId(), genRRefId());
}

template std::shared_ptr<UserRRef<IValue>>
    RRefContext::createUserRRef<IValue>(worker_id_t ownerId);

template std::shared_ptr<UserRRef<py::object>>
    RRefContext::createUserRRef<py::object>(worker_id_t ownerId);


template <typename T>
std::shared_ptr<UserRRef<T>> RRefContext::createUserRRef(
    worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId) {
  TORCH_CHECK(
      ownerId != getWorkerId(), "RRef owner cannot create user RRef.");
  // RRefContext does not track user RRefs, it will be destructed when there
  // is no shared_ptrs pointing to it. NB: cannot use make_shared here as the
  // constructor of UserRRef is private
  auto userRRef =
      std::shared_ptr<UserRRef<T>>(new UserRRef<T>(ownerId, rrefId, forkId));

  {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(
        pendingUsers_.find(forkId) == pendingUsers_.end(),
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

template std::shared_ptr<UserRRef<IValue>>
    RRefContext::createUserRRef<IValue>(
        worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId);

template std::shared_ptr<UserRRef<py::object>>
    RRefContext::createUserRRef<py::object>(
        worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId);


template <typename T>
std::shared_ptr<RRef> RRefContext::getOrCreateRRef(at::IValue&& value) {
  auto rfd = RRefForkData::fromIValue(std::move(value));
  return getOrCreateRRef<T>(rfd.ownerId_, rfd.rrefId_, rfd.forkId_);
}

template std::shared_ptr<RRef>
    RRefContext::getOrCreateRRef<IValue>(at::IValue&& value);

template std::shared_ptr<RRef>
    RRefContext::getOrCreateRRef<py::object>(at::IValue&& value);


template <typename T>
std::shared_ptr<RRef> RRefContext::getOrCreateRRef(
    worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId) {
  if (ownerId == getWorkerId()) {
    return getOrCreateOwnerRRef<T>(rrefId);
  } else {
    return createUserRRef<T>(ownerId, rrefId, forkId);
  }
}

template std::shared_ptr<RRef>
    RRefContext::getOrCreateRRef<IValue>(
        worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId);

template std::shared_ptr<RRef>
    RRefContext::getOrCreateRRef<py::object>(
        worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId);


template <typename T>
std::shared_ptr<OwnerRRef<T>> RRefContext::getOrCreateOwnerRRef(
    const RRefId& rrefId) {
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

template std::shared_ptr<OwnerRRef<IValue>>
    RRefContext::getOrCreateOwnerRRef<IValue>(const RRefId& rrefId);

template std::shared_ptr<OwnerRRef<py::object>>
    RRefContext::getOrCreateOwnerRRef<py::object>(const RRefId& rrefId);


RRefForkData RRefContext::forkTo(
    const std::shared_ptr<RRef>& rref,
    worker_id_t forkDst) {
  auto forkRequest = rref->fork();
  if (rref->owner() != forkDst) {
    // if fork destination if not owner, the forked UserRRef needs to be tracked
    // properly
    if (rref->isOwner()) {
      // fork from owner
      agent_->send(
          agent_->getWorkerId(forkDst),
          acceptUserRRef(forkRequest.rrefId_, forkRequest.forkId_));
    } else {
      // fork from user, rref cannot be destructed until the fork request is
      // accepted by the owner
      {
        std::lock_guard<std::mutex> lock(mutex_);
        pendingForkRequests_[forkRequest.forkId_] = rref;
      }
      // notify owner
      agent_->send(
          agent_->getWorkerId(rref->owner()),
          ScriptForkNotify(
              forkRequest.ownerId_,
              forkRequest.rrefId_,
              forkRequest.forkId_,
              forkDst
          ).toMessage());
    }
  }
  return forkRequest;
}

Message RRefContext::acceptUserRRef(const RRefId& rrefId, const ForkId& forkId) {
  addForkOfOwner(rrefId, forkId);
  return ScriptUserAccept(getWorkerId(), rrefId, forkId).toMessage();
}

Message RRefContext::acceptForkRequest(
    const RRefId& rrefId, const ForkId& forkId, worker_id_t forkDst) {
  agent_->send(
      agent_->getWorkerId(forkDst),
      acceptUserRRef(rrefId, forkId));
  // notify fork caller UserRRef
  return ScriptForkAccept(forkId).toMessage();
}

void RRefContext::finishForkRequest(const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = pendingForkRequests_.find(forkId);
  AT_ASSERT(
      iter != pendingForkRequests_.end(),
      "Cannot finish a non-exist fork request.");
  pendingForkRequests_.erase(iter);
}

void RRefContext::finishUserRRef(const RRefId& rrefId, const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(
      pendingAcceptedUsers_.find(forkId) == pendingAcceptedUsers_.end(),
      "Inconsistent state, attempt to accept the same UserRRef twice.")

  auto iter = pendingUsers_.find(forkId);
  if (iter != pendingUsers_.end()) {
    TORCH_INTERNAL_ASSERT(iter->second->id() == rrefId,
        "Attempt to accept a fork with incorrect RRefId.");
    // UserRRef created before receiving RREF_USER_ACCEPT message
    pendingUsers_.erase(iter);
  } else {
    // RREF_USER_ACCEPT arrives before UserRRef is created, remove it
    pendingAcceptedUsers_.insert(forkId);
  }
}

void RRefContext::addForkOfOwner(const at::IValue& value) {
  auto rfd = RRefForkData::fromIValue(value);
  AT_ASSERT(
      rfd.ownerId_ == getWorkerId(),
      "RRef user should never receive fork notification.");
  addForkOfOwner(rfd.rrefId_, rfd.forkId_);
}

void RRefContext::addForkOfOwner(const RRefId& rrefId, const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& rrefForks = forks_[rrefId];
  AT_ASSERT(
      rrefForks.find(forkId) == rrefForks.end(),
      "Got fork notification twice on the same RRef ",
      forkId);
  rrefForks.insert(forkId);
}

void RRefContext::delForkOfOwner(const at::IValue& value) {
  auto rfd = RRefForkData::fromIValue(value);
  AT_ASSERT(
      rfd.ownerId_ == getWorkerId(),
      "RRef user should never receive delete notification.");
  delForkOfOwner(rfd.rrefId_, rfd.forkId_);
}

void RRefContext::delForkOfOwner(const RRefId& rrefId, const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& rrefForks = forks_[rrefId];
  AT_ASSERT(
      rrefForks.find(forkId) != rrefForks.end(),
      "Attempt to delete a non-exist fork ",
      forkId);
  rrefForks.erase(rrefId);

  if (rrefForks.empty()) {
    owners_.erase(rrefId);
    forks_.erase(rrefId);
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
