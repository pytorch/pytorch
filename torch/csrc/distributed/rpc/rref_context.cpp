#include <torch/csrc/distributed/rpc/rref_context.h>

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

RRefId RRefContext::genRRefId() {
  return RRefId(getWorkerId(), nextLocalId_++);
}

const std::shared_ptr<RpcAgent>& RRefContext::agent() const {
  return agent_;
}

void RRefContext::addFork(const at::IValue& value) {
  auto rfd = RRefForkData::fromIValue(value);
  AT_ASSERT(
      rfd.ownerId_ == getWorkerId(),
      "RRef user should never receive fork notification.");
  std::lock_guard<std::mutex> lock(mutex_);
  auto& rrefForks = forks_[rfd.rrefId_];
  AT_ASSERT(
      rrefForks.find(rfd.forkId_) == rrefForks.end(),
      "Got fork notification twice on the same RRef ",
      rfd.rrefId_);
  rrefForks.insert(rfd.forkId_);
}

void RRefContext::delFork(const at::IValue& value) {
  auto rfd = RRefForkData::fromIValue(value);
  AT_ASSERT(
      rfd.ownerId_ == getWorkerId(),
      "RRef user should never receive delete notification.");
  std::lock_guard<std::mutex> lock(mutex_);
  auto& rrefForks = forks_[rfd.rrefId_];
  AT_ASSERT(
      rrefForks.find(rfd.forkId_) != rrefForks.end(),
      "Attempt to delete a non-exist fork ",
      rfd.forkId_);
  rrefForks.erase(rfd.forkId_);
  if (rrefForks.empty()) {
    owners_.erase(rfd.rrefId_);
    forks_.erase(rfd.rrefId_);
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
