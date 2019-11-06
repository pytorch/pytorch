#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>

#include <sstream>

namespace torch {
namespace distributed {
namespace rpc {

RRefContext& RRefContext::getInstance() {
  static RRefContext context(RpcAgent::getDefaultRpcAgent());
  return context;
}

void RRefContext::destroyInstance() {
  RRefContext::getInstance().checkRRefLeaks();
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
  if (!owners_.empty()) {
    AutoGIL ag;
    owners_.clear();
  }
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
  // is no shared_ptrs pointing to it.
  //
  // NB: cannot use make_shared here as the constructor of UserRRef is private.
  // NB: This UserRRef has not been confirmed by the owner yet. This function's
  // call site is responsible for adding this UserRRef to pendingUsers_.
  // Currently, there are two call sites.
  // (1) The creator user in python_functions.cpp
  // (2) The callee user in RRefContext::notifyOwnerAndParentOfFork.
  //
  // The reason for not adding the pending user here is to put addPendingUser()
  // close to where the RPC occurs, and it is more clear to pair it with
  // deletePendingUser() in the response callback at the call site.
  return std::shared_ptr<UserRRef<T>>(new UserRRef<T>(ownerId, rrefId, forkId));
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
    //
    // NB: cannot use make_shared here as the constructor of OwnerRRef is
    // private.
    auto rref =
        std::shared_ptr<OwnerRRef<T>>(new OwnerRRef<T>(getWorkerId(), rrefId));
    owners_[rref->rrefId()] = rref;
    return rref;

  } else {
    // Scenario (2) retrieving an existing RRef
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
    // Note [Early Fork Registration]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // If the parent (caller) is the owner, directly register the fork, instead
    // of waiting for another RREF_FORK_REQUEST or RREF_CHILD_ACCEPT message. An
    // Alternative is adding the fork when the callee user ACKs. However, before
    // that, the owner still have to adds the OwnerRRef into some map to keep it
    // alive (e.g., in pendingChildren_). Hence, adding the fork here or in the
    // ACK does not making any difference but only add complexity.
    // TODO: When adding failure retries and timeout, this fork needs to be
    // deleted if the owner does not receive the ACK within the timeout.
    addForkOfOwner(rfd.rrefId_, rfd.forkId_);
  } else {
    // Note [Useful Phantom Fork ID for User to Owner Call]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // If the callee of dist.remote or dist.rpc is the owner of this RRef, the
    // callee will not create a fork using this rfd.forkId_, because the owner
    // will only keep one `OwnerRRef` instance and will not create any
    // `UserRRef` instances. However, this rfd.forkId_ is still necessary, as
    // the caller user needs to keep this `UserRRef` alive until it gets the
    // ACK from the callee owner. Otherwise, the delete message could arrive
    // at the owner before this dist.rpc or dist.remote call, which could
    // potentially trigger the `OwnerRRef` to be deleted before running the
    // user code.
    addPendingChild(rfd.forkId_, rref);
  }
  return rfd;
}

void RRefContext::notifyOwnerAndParentOfFork(
    const ForkId& forkId,
    worker_id_t parent,
    const std::shared_ptr<RRef>& rref) {
  if (parent == rref->owner()) {
    // If the parent is the owner, this fork has already been added into the
    // forks_ map when the owner sends the message to the callee user. Hence,
    // it is not necessary to send another RREF_CHILD_ACCEPT or
    // RREF_FORK_REQUEST back to the owner. See Note [Early Fork Registration].
    return;
  }

  if (rref->isOwner()) {
    // See Note [Useful Phantom Fork ID for User to Owner Call]
    // In this case, the owner is the caller, and it does not add the fork id
    // into forks_. Because, there will be no real `UserRRef` associated with
    // this fork ID.
    auto fm = agent_->send(
        agent_->getWorkerInfo(parent), RRefChildAccept(forkId).toMessage());
    fm->addCallback([](const Message& message) { handleException(message); });
  } else {
    auto fm = agent_->send(
        agent_->getWorkerInfo(rref->owner()),
        RRefForkRequest(rref->rrefId(), forkId).toMessage());

    addPendingUser(forkId, rref);
    fm->addCallback([this, forkId, parent](const Message& message) {
      handleException(message);
      this->finishForkRequest(forkId, parent);
    });
  }
}

void RRefContext::addPendingChild(
    const ForkId& forkId,
    const std::shared_ptr<RRef>& rref) {
  // see Note [Early Fork Registration]
  // If the parent is the owner, it should directly add the child UserRRef as a
  // fork.
  TORCH_INTERNAL_ASSERT(
      !rref->isOwner(), "OwnerRRef should not have a pending child.");
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_INTERNAL_ASSERT(
      pendingChildren_.find(forkId) == pendingChildren_.end(),
      "Inconsistent states: attempt to add the same child fork twice.");
  pendingChildren_[forkId] = rref;
}

void RRefContext::delPendingChild(const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = pendingChildren_.find(forkId);
  TORCH_INTERNAL_ASSERT(
      iter != pendingChildren_.end(),
      "Inconsistent states: attempt to delete a non-exist child fork.");
  pendingChildren_.erase(iter);
}

void RRefContext::addPendingUser(
    const ForkId& forkId,
    const std::shared_ptr<RRef>& rref) {
  TORCH_INTERNAL_ASSERT(
      !rref->isOwner(), "Attempt to add an OwnerRRef as a pending User.");
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_INTERNAL_ASSERT(
      pendingUsers_.find(forkId) == pendingUsers_.end(),
      "Inconsistent states: attempt to add the same UserRRef twice.");
  pendingUsers_[forkId] = rref;
}

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
        std::cout << agent_->getWorkerInfo().id_ << " ==== deleting owner " << rrefId << std::endl << std::flush;
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
