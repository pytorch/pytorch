#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>

#include <sstream>

namespace torch {
namespace distributed {
namespace rpc {

std::unique_ptr<RRefContext> RRefContext::context_ = nullptr;

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

void RRefContext::handleException(const Message& message) {
  if (message.type() == MessageType::EXCEPTION) {
    // TODO: allow users to register an error handler and call it here.
    std::string err(message.payload().begin(), message.payload().end());
    VLOG(1) << "Got exception: " << err << std::endl << std::flush;
    throw std::runtime_error(err);
  }
}

RRefContext::RRefContext(std::shared_ptr<RpcAgent> agent)
    : agent_(std::move(agent)) {}

RRefContext::~RRefContext() {
  AutoGIL ag;
  owners_.clear();
}

void RRefContext::checkRRefLeaks() {
  if (!forks_.empty()) {
    std::stringstream ss;
    for (auto& entry : forks_) {
      const RRefId& rrefId = entry.first;
      for (const auto& forkId : entry.second) {
        ss << "Leaking RRef " << rrefId << " with fork Id " << forkId
           << std::endl;
      }
    }
    AT_ERROR(ss.str());
  }
}

template <typename T>
std::shared_ptr<UserRRef<T>> RRefContext::createUserRRef(worker_id_t ownerId) {
  TORCH_CHECK(ownerId != getWorkerId(), "Cannot create UserRRef on owner.");
  return createUserRRef<T>(
      ownerId, genGloballyUniqueId(), genGloballyUniqueId());
}

template std::shared_ptr<UserRRef<IValue>> RRefContext::createUserRRef<IValue>(
    worker_id_t ownerId);

template std::shared_ptr<UserRRef<py::object>> RRefContext::createUserRRef<
    py::object>(worker_id_t ownerId);

template <typename T>
std::shared_ptr<UserRRef<T>> RRefContext::createUserRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId) {
  TORCH_CHECK(ownerId != getWorkerId(), "RRef owner cannot create user RRef.");
  // RRefContext does not track user RRefs, it will be destructed when there
  // is no shared_ptrs pointing to it. NB: cannot use make_shared here as the
  // constructor of UserRRef is private
  auto userRRef =
      std::shared_ptr<UserRRef<T>>(new UserRRef<T>(ownerId, rrefId, forkId));
  if (forkId.createdOn_ != ownerId) {
    addPendingUser(forkId, userRRef);
  }
  return userRRef;
}

template std::shared_ptr<UserRRef<IValue>> RRefContext::createUserRRef<IValue>(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId);

template std::shared_ptr<UserRRef<py::object>> RRefContext::createUserRRef<
    py::object>(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId);

template <typename T>
std::shared_ptr<RRef> RRefContext::getOrCreateRRef(const RRefForkData& rfd) {
  auto& ownerId = rfd.ownerId_;
  auto& rrefId = rfd.rrefId_;
  auto& forkId = rfd.forkId_;
  if (ownerId == getWorkerId()) {
    return getOrCreateOwnerRRef<T>(rrefId);
  } else {
    return createUserRRef<T>(ownerId, rrefId, forkId);
  }
}

template std::shared_ptr<RRef> RRefContext::getOrCreateRRef<IValue>(
    const RRefForkData& rfd);

template std::shared_ptr<RRef> RRefContext::getOrCreateRRef<py::object>(
    const RRefForkData& rfd);

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
    auto rref =
        std::shared_ptr<OwnerRRef<T>>(new OwnerRRef<T>(getWorkerId(), rrefId));
    owners_[rref->rrefId()] = rref;
    return rref;

  } else {
    // Scenario (3) retrieving an existing RRef
    return std::static_pointer_cast<OwnerRRef<T>>(iter->second);
  }
}

template std::shared_ptr<OwnerRRef<IValue>> RRefContext::getOrCreateOwnerRRef<
    IValue>(const RRefId& rrefId);

template std::shared_ptr<OwnerRRef<py::object>> RRefContext::
    getOrCreateOwnerRRef<py::object>(const RRefId& rrefId);

RRefForkData RRefContext::prepareChildFork(const std::shared_ptr<RRef>& rref) {
  auto rfd = rref->fork();
  if (rref->isOwner()) {
    addForkOfOwner(rfd.rrefId_, rfd.forkId_);
  } else {
    if (rref->isPyObj()) {
      addPendingChild(
          rfd.forkId_, std::static_pointer_cast<UserRRef<py::object>>(rref));
    } else {
      addPendingChild(
          rfd.forkId_, std::static_pointer_cast<UserRRef<IValue>>(rref));
    }
  }
  return rfd;
}

void RRefContext::notifyOwnerAndParentOfFork(
    const ForkId& forkId,
    worker_id_t parent,
    const std::shared_ptr<RRef>& rref) {
  if (parent != rref->owner()) {
    if (rref->isOwner()) {
      auto fm = agent_->send(
          agent_->getWorkerInfo(parent), RRefChildAccept(forkId).toMessage());
      fm->addCallback([](const Message& message) { handleException(message); });
    } else {
      auto fm = agent_->send(
          agent_->getWorkerInfo(rref->owner()),
          RRefForkRequest(rref->rrefId(), forkId).toMessage());

      fm->addCallback([this, forkId, parent](const Message& message) {
        handleException(message);
        this->finishForkRequest(forkId, parent);
      });
    }
  }
}

template <typename T>
void RRefContext::addPendingChild(
    const ForkId& forkId,
    const std::shared_ptr<UserRRef<T>>& rref) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_INTERNAL_ASSERT(
      pendingChildren_.find(forkId) == pendingChildren_.end(),
      "Inconsistent states: attempt to add the same child fork twice.");
  pendingChildren_[forkId] = rref;
}

template void RRefContext::addPendingChild<IValue>(
    const ForkId& forkId,
    const std::shared_ptr<UserRRef<IValue>>& rref);

template void RRefContext::addPendingChild<py::object>(
    const ForkId& forkId,
    const std::shared_ptr<UserRRef<py::object>>& rref);

void RRefContext::delPendingChild(const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = pendingChildren_.find(forkId);
  TORCH_INTERNAL_ASSERT(
      iter != pendingChildren_.end(),
      "Inconsistent states: attempt to delete a non-exist child fork.");
  pendingChildren_.erase(iter);
}

template <typename T>
void RRefContext::addPendingUser(
    const ForkId& forkId,
    const std::shared_ptr<UserRRef<T>>& rref) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_INTERNAL_ASSERT(
      pendingUsers_.find(forkId) == pendingUsers_.end(),
      "Inconsistent states: attempt to add the same UserRRef twice.");
  pendingUsers_[forkId] = rref;
}

template void RRefContext::addPendingUser<IValue>(
    const ForkId& forkId,
    const std::shared_ptr<UserRRef<IValue>>& rref);

template void RRefContext::addPendingUser<py::object>(
    const ForkId& forkId,
    const std::shared_ptr<UserRRef<py::object>>& rref);

void RRefContext::delPendingUser(const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = pendingUsers_.find(forkId);
  TORCH_INTERNAL_ASSERT(
      iter != pendingUsers_.end(),
      "Inconsistent states: attempt to delete a non-exist UserRRef.");
  pendingUsers_.erase(iter);
}

void RRefContext::finishForkRequest(const ForkId& forkId, worker_id_t parent) {
  delPendingUser(forkId);
  auto fm = agent_->send(
      agent_->getWorkerInfo(parent), RRefChildAccept(forkId).toMessage());

  fm->addCallback([](const Message& message) { handleException(message); });
}

void RRefContext::addForkOfOwner(const RRefId& rrefId, const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& rrefForks = forks_[rrefId];
  TORCH_INTERNAL_ASSERT(
      rrefForks.find(forkId) == rrefForks.end(),
      "Got fork notification twice on the same RRef ",
      forkId);
  rrefForks.insert(forkId);
}

void RRefContext::delForkOfOwner(const RRefId& rrefId, const ForkId& forkId) {
  std::shared_ptr<RRef> deletedRRef = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto rrefIter = forks_.find(rrefId);
    TORCH_INTERNAL_ASSERT(
        rrefIter != forks_.end(),
        "Inconsistent states, deleting a fork before the owner knows it.");
    auto& rrefForks = rrefIter->second;
    auto forkIter = rrefForks.find(forkId);
    TORCH_INTERNAL_ASSERT(
        forkIter != rrefForks.end(),
        "Attempt to delete a non-exist fork ",
        forkId);

    rrefForks.erase(forkId);

    if (rrefForks.empty()) {
      auto ownerIter = owners_.find(rrefId);
      if (ownerIter != owners_.end()) {
        deletedRRef = ownerIter->second;
        owners_.erase(ownerIter);
      }
      forks_.erase(rrefIter);
    }
  }
  if (deletedRRef && deletedRRef->isPyObj()) {
    AutoGIL ag;
    deletedRRef.reset();
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
