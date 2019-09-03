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
  TORCH_CHECK(RRefContext::context_,
      "Have to initialize RRefContext before use.");
  return RRefContext::context_;
}

RRefContext::RRefContext(std::shared_ptr<RpcAgent> agent)
    : agent_(std::move(agent)) {
}

worker_id_t RRefContext::getWorkerId() const {
  return agent_->getWorkerId().id_;
}

RRefId RRefContext::genRRefId() {
  return RRefId(getWorkerId(), nextLocalId_++);
}

const std::shared_ptr<RpcAgent>& RRefContext::agent() const {
  return agent_;
}

RRefForkData RRefContext::forkTo(std::shared_ptr<RRef> rref, worker_id_t forkDst) {
  auto forkRequest = rref->fork();
  if (rref->owner() != forkDst) {
    // if fork destination if not owner, the forked UserRRef needs to be tracked
    // properly
    if (rref->isOwner()) {
      // fork from owner
      acceptUserRRef(forkRequest.rrefId_, forkRequest.forkId_, forkDst);
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
          ScriptForkNotify(forkRequest.toIValue(), forkDst).toMessage()
      );
    }
  }
  return forkRequest;
}

void RRefContext::acceptUserRRef(
    const RRefId& rrefId, const ForkId& forkId, worker_id_t user) {
  addForkOfOwner(rrefId, forkId);
  agent_->send(
      agent_->getWorkerId(user),
      ScriptUserAccept(forkId.toIValue()).toMessage()
  );
}

void RRefContext::acceptForkRequest(IValue value, worker_id_t forkDst) {
  auto forkRequest = RRefForkData::fromIValue(std::move(value));
  auto& rrefId = forkRequest.rrefId_;
  auto& forkId = forkRequest.forkId_;
  acceptUserRRef(rrefId, forkId, forkDst);
  // notify fork caller UserRRef
  agent_->send(
      agent_->getWorkerId(forkId.createdOn_),
      ScriptForkAccept(forkId.toIValue()).toMessage()
  );
}

void RRefContext::finishForkRequest(IValue value) {
  auto forkRequest = RRefForkData::fromIValue(std::move(value));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = pendingForkRequests_.find(forkRequest.forkId_);
    AT_ASSERT(iter != pendingForkRequests_.end(),
        "Cannot finish a non-exist fork request.");
    pendingForkRequests_.erase(iter);
  }
}

void RRefContext::finishUserRRef(IValue value) {
  auto forkId = ForkId::fromIValue(std::move(value));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(
        pendingAcceptedUsers_.find(forkId) == pendingAcceptedUsers_.end(),
        "Inconsistent state, attempt to accept the same UserRRef twice.")

    auto iter = pendingUsers_.find(forkId);
    if (iter != pendingUsers_.end()) {
      // UserRRef created before receiving RREF_USER_ACCEPT message
      pendingUsers_.erase(iter);
    } else {
      // RREF_USER_ACCEPT arrives before UserRRef is created, remove it
      pendingAcceptedUsers_.insert(std::move(forkId));
    }
  }
}

void RRefContext::addForkOfOwner(at::IValue&& value) {
  auto rfd = RRefForkData::fromIValue(std::move(value));
  AT_ASSERT(rfd.ownerId_ == getWorkerId(),
      "RRef user should never receive fork notification.");
  addForkOfOwner(rfd.rrefId_, rfd.forkId_);
}

void RRefContext::addForkOfOwner(const RRefId& rrefId, const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& rrefForks = forks_[rrefId];
  AT_ASSERT(rrefForks.find(forkId) == rrefForks.end(),
      "Got fork notification twice on the same RRef ", forkId);
  rrefForks.insert(forkId);
}

void RRefContext::delForkOfOwner(at::IValue&& value) {
  auto rfd = RRefForkData::fromIValue(std::move(value));
  AT_ASSERT(rfd.ownerId_ == getWorkerId(),
      "RRef user should never receive delete notification.");
  delForkOfOwner(rfd.rrefId_, rfd.forkId_);
}

void RRefContext::delForkOfOwner(const RRefId& rrefId, const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& rrefForks = forks_[rrefId];
  AT_ASSERT(rrefForks.find(forkId) != rrefForks.end(),
      "Attempt to delete a non-exist fork ", forkId);
  rrefForks.erase(rrefId);
  if (rrefForks.empty()) {
    owners_.erase(rrefId);
    forks_.erase(rrefId);
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
