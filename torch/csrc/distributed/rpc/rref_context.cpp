#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>

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
    auto ownerRRef = getOrCreateOwnerRRef<T>(rrefId);
    // See Note [Fork Request]
    std::lock_guard<std::mutex> lock(mutex_);
    auto rrefIter = forks_.find(rrefId);
    // Although we know that someone (either a UserRRef or a OwnerRRef) has
    // sent a fork request RREF_FORK_NOTIFY from somewhere, this could still be
    // the first time the owner sees this rrefId, as there is no order
    // guarantee on message delivery. Hence, the owner might not know about the
    // forkId either. However, we know that (1) there will be an
    // RREF_FORK_NOTIFY message arriving in the future, or (2) the message might
    // have already arrived.
    if (rrefIter != forks_.end() &&
        rrefIter->second.find(forkId) != rrefIter->second.end()) {
      // scenario (2): fork request arrived before rpc/remote request/response
      delForkOfOwnerNoLock(rrefId, forkId);
    } else {
      // scenario (1): fork request will arrive after rpc/remote
      // request/response
      expectingForkReqeusts_.insert(forkId);
    }
    return ownerRRef;
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
    return std::dynamic_pointer_cast<OwnerRRef<T>>(iter->second);
  }
}

template std::shared_ptr<OwnerRRef<IValue>> RRefContext::getOrCreateOwnerRRef<
    IValue>(const RRefId& rrefId);

template std::shared_ptr<OwnerRRef<py::object>> RRefContext::
    getOrCreateOwnerRRef<py::object>(const RRefId& rrefId);

RRefForkData RRefContext::forkTo(
    const std::shared_ptr<RRef>& rref,
    worker_id_t forkDst) {
  auto forkRequest = rref->fork();
  // Note [Fork Request]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  // Forked UserRRef needs to be tracked properly, regardless if the destination
  // is the owner or not. If the destination is a user, it is obvious that, the
  // RRef needs to be kept alive on owner. If the destination is the owner, we
  // still need the fork request. Because:
  //
  // (1) As we use ThreadPool on both sender and receiver, there is no guarantee
  //     on message delivery order. It is possible that the delete message is
  //     processed before some earlier rpc/remote calls which use this RRef as
  //     an argument, and the delete message might have triggered the deletion
  //     of the OwnerRRef.
  //
  // (2) Similar problem exist if the RRef is involved in the response from a
  //     user to the owner.
  //
  // Therefore, the RRefForkNotify message is always sent no matter if the owner
  // is the destination or not. If the owner is the destination, the owner will
  // not create any UserRRefs using the ForkId. Instead, it only adds the ForkId
  // into forks_, which will later be dropped in getOrCreateRRef(...).
  // Otherwise, if the destination is a user, the callee user will use the
  // ForkId to create a UserRRef and that UserRRef will control the lifetime of
  // the ForkId on owner.
  if (rref->isOwner()) {
    // fork from owner
    auto fm = agent_->send(
        agent_->getWorkerInfo(forkDst),
        acceptUserRRef(forkRequest.rrefId_, forkRequest.forkId_));

    fm->addCallback([](const Message& message) { handleException(message); });
  } else {
    // fork from user, rref cannot be destructed until the fork request is
    // accepted by the owner
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pendingForkRequests_[forkRequest.forkId_] = rref;
    }
    // notify owner
    auto fm = agent_->send(
        agent_->getWorkerInfo(rref->owner()),
        RRefForkNotify(forkRequest.rrefId_, forkRequest.forkId_, forkDst)
            .toMessage());

    fm->addCallback([this](const Message& message) {
      handleException(message);
      auto rfa = RRefForkAccept::fromMessage(message);
      this->finishForkRequest(rfa.forkId());
    });
  }
  return forkRequest;
}

Message RRefContext::acceptUserRRef(
    const RRefId& rrefId,
    const ForkId& forkId) {
  addForkOfOwner(rrefId, forkId);
  return RRefUserAccept(rrefId, forkId).toMessage();
}

Message RRefContext::acceptForkRequest(
    const RRefId& rrefId,
    const ForkId& forkId,
    const worker_id_t forkDst) {
  if (forkDst == getWorkerId()) {
    // forking to the owner
    // See Note [Fork Request]
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = expectingForkReqeusts_.find(forkId);
    if (iter == expectingForkReqeusts_.end()) {
      // fork request arrives before the rpc/remote request/response
      addForkOfOwnerNoLock(rrefId, forkId);
    } else {
      // rpc/remote request/response arrives before the fork request
      expectingForkReqeusts_.erase(iter);
    }
  } else {
    // forking to a user
    auto fm = agent_->send(
        agent_->getWorkerInfo(forkDst), acceptUserRRef(rrefId, forkId));

    fm->addCallback([](const Message& message) { handleException(message); });
  }
  // notify fork caller UserRRef
  return RRefForkAccept(forkId).toMessage();
}

void RRefContext::finishForkRequest(const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = pendingForkRequests_.find(forkId);
  TORCH_INTERNAL_ASSERT(
      iter != pendingForkRequests_.end(),
      "Cannot finish a non-exist fork request.");
  pendingForkRequests_.erase(iter);
}

void RRefContext::finishUserRRef(const RRefId& rrefId, const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_INTERNAL_ASSERT(
      pendingAcceptedUsers_.find(forkId) == pendingAcceptedUsers_.end(),
      "Inconsistent state, attempt to accept the same UserRRef twice.")

  auto iter = pendingUsers_.find(forkId);
  if (iter != pendingUsers_.end()) {
    TORCH_INTERNAL_ASSERT(
        iter->second->rrefId() == rrefId,
        "Attempt to accept a fork with incorrect RRefId.");
    // UserRRef created before receiving RREF_USER_ACCEPT message
    pendingUsers_.erase(iter);
  } else {
    // RREF_USER_ACCEPT arrives before UserRRef is created, remove it
    pendingAcceptedUsers_.insert(forkId);
  }
}

void RRefContext::addForkOfOwner(const RRefId& rrefId, const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  addForkOfOwnerNoLock(rrefId, forkId);
}

void RRefContext::delForkOfOwner(const RRefId& rrefId, const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  delForkOfOwnerNoLock(rrefId, forkId);
}

void RRefContext::addForkOfOwnerNoLock(
    const RRefId& rrefId,
    const ForkId& forkId) {
  auto& rrefForks = forks_[rrefId];
  TORCH_INTERNAL_ASSERT(
      rrefForks.find(forkId) == rrefForks.end(),
      "Got fork notification twice on the same RRef ",
      forkId);
  rrefForks.insert(forkId);
}

void RRefContext::delForkOfOwnerNoLock(
    const RRefId& rrefId,
    const ForkId& forkId) {
  auto iter = forks_.find(rrefId);
  TORCH_INTERNAL_ASSERT(
      iter != forks_.end(),
      "Inconsistent states, deleting a fork before the owner knows it.");
  auto& rrefForks = iter->second;
  TORCH_INTERNAL_ASSERT(
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
