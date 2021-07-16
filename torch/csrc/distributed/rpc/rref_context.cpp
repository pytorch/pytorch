#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/utils.h>

#include <sstream>

namespace torch {
namespace distributed {
namespace rpc {

thread_local std::vector<std::shared_ptr<RRefContext::PendingUserState>>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    RRefContext::userTable_;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local bool RRefContext::recording_ = false;

namespace callback {
void confirmPendingUser(
    const JitFuture& jitFuture,
    const ForkId& expectedForkId) {
  if (!jitFuture.hasError()) {
    auto msgPtr = jitFuture.constValue().toCustomClass<Message>();
    auto msgType = msgPtr->type();
    auto rpc = deserializeResponse(*msgPtr, msgType);
    auto rr = dynamic_cast<RemoteRet*>(rpc.get());
    TORCH_INTERNAL_ASSERT(rr->forkId() == expectedForkId);
  } else {
    // Handle errors, such as timeouts, by invoking the error handler on the
    // rref.
    // Note [Best Effort Error handling for Remote calls]:
    // When remote calls initiated by rpc.remote() fail, such as with a timeout
    // error, we take a best-effort approach to error handling. We handle errors
    // when callbacks corresponding to the remote call run, and set the error
    // information on the RRef. If the RRef has not been used by the application
    // before this process (such as to_here or fork call), then future uses of
    // the RRef will appropriately raise errors. However, it is possible that
    // the user application will use the RRef before the errors are handled. In
    // this case, errors may not be raised as they have not yet been handled.
    auto rref_ptr = RRefContext::getInstance().getPendingUser(expectedForkId);
    auto errorType = getRPCErrorType(jitFuture);
    rref_ptr->handleError(errorType, jitFuture);
  }
  RRefContext::getInstance().delPendingUser(expectedForkId);
}

c10::intrusive_ptr<RRef> finishCreatingOwnerRRef(
    const JitFuture& jitFuture,
    const RRefId& rrefId) {
  if (jitFuture.hasError()) {
    auto& ctx = RRefContext::getInstance();
    // We expect to run this callback only after the OwnerRRef has been created,
    // since this is only invoked when sending to self.
    auto rref_ptr =
        fromRRefInterface(ctx.getOwnerRRef(rrefId, /* foreCreated */ true)
                              ->constValue()
                              .toRRef());
    auto errorType = getRPCErrorType(jitFuture);
    rref_ptr->handleError(errorType, jitFuture);
    // OwnerRRefs do not have a forkId, so don't need to assert here.
    auto deletedRRef =
        ctx.delForkOfOwner(rref_ptr->rrefId(), rref_ptr->rrefId());
    return deletedRRef;
  } else {
    auto msgPtr = jitFuture.constValue().toCustomClass<Message>();
    auto msgType = msgPtr->type();
    auto rpc = deserializeResponse(*msgPtr, msgType);
    auto rr = dynamic_cast<RemoteRet*>(rpc.get());
    TORCH_INTERNAL_ASSERT(
        rr->rrefId() == rr->forkId(),
        "Expecting an OwnerRRef as RemoteRet but got a fork.");
    auto& ctx = RRefContext::getInstance();
    auto deletedRRef = ctx.delForkOfOwner(rr->rrefId(), rr->rrefId());
    return deletedRRef;
  }
}

} // namespace callback

// Keys for RRef-related debug information.
const std::string kNumOwnerRRefs = "num_owner_rrefs";
const std::string kNumPendingFutures = "num_pending_futures";
const std::string kNumPendingUsers = "num_pending_users";
const std::string kNumForks = "num_forks";

RRefContext& RRefContext::getInstance() {
  // Leaky singleton to avoid module destructor races.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static RRefContext* context = new RRefContext(RpcAgent::getCurrentRpcAgent());
  return *context;
}

std::vector<c10::intrusive_ptr<RRef>> RRefContext::destroyInstance(
    bool ignoreRRefLeak) {
  auto& ctx = RRefContext::getInstance();
  {
    std::lock_guard<std::mutex> lock(ctx.destroyedMutex_);
    ctx.destroyed_ = true;
  }
  ctx.checkRRefLeaks(ignoreRRefLeak);
  std::vector<c10::intrusive_ptr<RRef>> deletedRRefs;
  for (auto& entry : ctx.owners_) {
    auto rref = entry.second;
    if (rref->isPyObj()) {
      deletedRRefs.emplace_back(std::move(rref));
    }
  }
  ctx.owners_.clear();
  ctx.pendingOwners_.clear();
  return deletedRRefs;
}

void RRefContext::handleException(const JitFuture& jitFuture) {
  if (jitFuture.hasError()) {
    auto errMsg = jitFuture.tryRetrieveErrorMessage();
    VLOG(1) << "Got exception: " << errMsg;
    TORCH_CHECK(false, errMsg);
  }
}

RRefContext::RRefContext(std::shared_ptr<RpcAgent> agent)
    : agent_(std::move(agent)), destroyed_(false) {}

RRefContext::~RRefContext() {
  if (!owners_.empty()) {
    VLOG(1) << "Destructing RRefContext with non-empty OwnerRRef set. "
            << "This would likely cause Python deref error. "
            << "Make sure destroyInstance() is invoked before destruction.";
  }
}

std::unordered_map<std::string, std::string> RRefContext::getDebugInfo() {
  std::unordered_map<std::string, std::string> info;
  std::unique_lock<std::mutex> lock(mutex_);
  auto ownerSize = owners_.size();
  auto numPendingUsers = pendingUsers_.size();
  int numForks = 0;
  for (const auto& owner : forks_) {
    numForks += owner.second.size();
  }
  lock.unlock();
  info[kNumOwnerRRefs] = c10::to_string(ownerSize);
  info[kNumPendingFutures] = c10::to_string(numPendingFutures_.load());
  info[kNumPendingUsers] = c10::to_string(numPendingUsers);
  info[kNumForks] = c10::to_string(numForks);
  return info;
}

void RRefContext::checkRRefLeaks(bool ignoreRRefLeak) {
  if (!forks_.empty()) {
    std::stringstream ss;
    for (auto& entry : forks_) {
      const RRefId& rrefId = entry.first;
      for (const auto& forkId : entry.second) {
        ss << "Leaking RRef " << rrefId << " with fork Id " << forkId
           << std::endl;
      }
    }

    LOG(WARNING)
        << "Detected RRef Leaks during shutdown. This usually "
        << "occurs when the application code still holds references to RRef "
        << "instances when calling shutdown(). If the program has "
        << "completed correctly and the process is exiting, it is OK to "
        << "ignore these leaks. However, if you program will keep running "
        << "after this, these leaks could result in memory leaks on RRef "
        << "owners. Please make sure all RRefs are out of scope and Python "
        << "GC has deleted them before calling shutdown(): \n"
        << ss.str();
    if (!ignoreRRefLeak) {
      TORCH_CHECK(false, ss.str());
    }
  }
}

c10::intrusive_ptr<UserRRef> RRefContext::createUserRRef(
    worker_id_t ownerId,
    const TypePtr& type) {
  TORCH_CHECK(ownerId != getWorkerId(), "Cannot create UserRRef on owner.");
  // Explicitly creating rrefId before forkId to make sure the order is
  // deterministic, as the argument evaluation order is system and compiler
  // dependent.
  const auto rrefId = genGloballyUniqueId();
  const auto forkId = genGloballyUniqueId();
  return createUserRRef(ownerId, rrefId, forkId, type);
}

c10::intrusive_ptr<UserRRef> RRefContext::createUserRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId,
    const TypePtr& type) {
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
  return c10::make_intrusive<UserRRef>(ownerId, rrefId, forkId, type);
}

void RRefContext::delUser(
    const worker_id_t owner,
    const RRefId& rrefId,
    const ForkId& forkId) {
  {
    std::lock_guard<std::mutex> lock(destroyedMutex_);
    if (!destroyed_) {
      // Sending an RRefUserDelete causes the receiver to run delForkOfOwner,
      // which is now idempotent. See the comment at RRefContext::delForkOfOwner
      // for more details.
      ++numPendingFutures_;
      auto jitFuture = agent_->sendWithRetries(
          agent_->getWorkerInfo(owner),
          RRefUserDelete(rrefId, forkId).toMessage());

      jitFuture->addCallback([this](JitFuture& future) {
        handleException(future);
        --numPendingFutures_;
      });
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  confirmedUsers_.erase(forkId);
}

void RRefContext::delAllUsersAndUnforkedOwners(
    std::chrono::milliseconds timeoutMillis) {
  // First, wait for all pending UserRRefs to be confirmed,
  // one kind is pendingUsers_, which are shared from Owner,
  // the other kind pendingChildren_, which are shared from another User.
  std::unordered_map<ForkId, c10::weak_intrusive_ptr<RRef>, ForkId::Hash>
      tempConfirmedUsers;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    bool noPending = deleteAllUsersCV_.wait_for(lock, timeoutMillis, [this]() {
      return pendingUsers_.size() == 0 && pendingChildren_.size() == 0;
    });
    if (!noPending) {
      LOG(ERROR)
          << "Timed out waiting for pending UserRRefs to be confirmed by owner and parent.";
    }
    tempConfirmedUsers.swap(confirmedUsers_);
  }

  // Start sending UserRRef delete messages, after all pendings are confirmed.
  // Note, there should be no new forkings in between, because it's assumed that
  // this utility is called during graceful shutdown, where no new user RPCs can
  // be initiaited anymore.
  for (const auto& user : tempConfirmedUsers) {
    c10::intrusive_ptr<RRef> rref_ptr = user.second.lock();
    if (!rref_ptr) {
      continue;
    }
    // tryDel() below will re-acquire lock, lock must be released here.
    rref_ptr->tryDel();
  }

  // If an rref in the owners_ map has never been forked, we will never get a
  // corresponding message from the forking node(s) telling us to delete the
  // RRef. Hence we delete the RRef here. This can occur when a remote call is
  // sent to self and times out.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<RRefId> unforkedOwners;
    for (const auto& it : owners_) {
      auto rrefId = it.first;
      if (forks_.find(rrefId) == forks_.end()) {
        // Successful fork of owner was never processed.
        unforkedOwners.push_back(rrefId);
      }
    }
    for (auto& rrefId : unforkedOwners) {
      LOG(INFO) << "Removing unforked OwnerRRef with RRefId: " << rrefId;
      auto iter = owners_.find(rrefId);
      TORCH_CHECK(
          iter != owners_.end(),
          c10::str("Did not find OwnerRRef with RRefId: ", rrefId));
      owners_.erase(iter);
    }
  }
  // Wait for this node to process all delete UserRRef messages it may get for
  // the OwnerRRefs that exist on this node.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    bool noOwner = deleteAllUsersCV_.wait_for(
        lock, timeoutMillis, [this]() { return owners_.size() == 0; });
    if (!noOwner) {
      LOG(ERROR) << "Timed out waiting for pending OwnerRRefs to be deleted.";
    }
  }
}

c10::intrusive_ptr<RRef> RRefContext::getOrCreateRRef(
    const RRefForkData& rrefForkData,
    const TypePtr& type) {
  auto& ownerId = rrefForkData.ownerId_;
  auto& rrefId = rrefForkData.rrefId_;
  auto& forkId = rrefForkData.forkId_;
  if (ownerId == getWorkerId()) {
    return getOrCreateOwnerRRef(rrefId, type);
  } else {
    return createUserRRef(ownerId, rrefId, forkId, type);
  }
}

c10::intrusive_ptr<OwnerRRef> RRefContext::getOrCreateOwnerRRef(
    const RRefId& rrefId,
    const TypePtr& type) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = owners_.find(rrefId);
  if (iter == owners_.end()) {
    // Scenario (1) the first time this owner knows about this RRef
    //
    // NB: cannot use make_shared here as the constructor of OwnerRRef is
    // private.
    auto rref = c10::make_intrusive<OwnerRRef>(
        getWorkerId(), rrefId, type, agent_->getDevices());
    owners_[rref->rrefId()] = rref;
    const auto pendingOwnerIter = pendingOwners_.find(rrefId);
    if (pendingOwnerIter != pendingOwners_.end()) {
      // cast to RRefInterface to hold it into IValue
      auto rrefPtr = fromOwnerRRef(rref);
      pendingOwnerIter->second->markCompleted(IValue(rrefPtr));
      pendingOwners_.erase(pendingOwnerIter);
    }
    return rref;
  } else {
    // Scenario (2) retrieving an existing RRef
    auto ownerRRef = fromRRefInterface(iter->second);
    // Now double check if the two types match
    //
    // Why we are special casing the check for tensor type here?
    // this is because tensor types might get specialized on tensors when
    // we pass inputs to the function, i.e. TensorType can filled with
    // specific shape info, requires_grad info, etc. so the OwerRRef we
    // found might already have those infos, but the `type` we passed in
    // here is a plain TensorType, they are not equal relationship:
    // specialized TensorType <: plain TensorType
    //
    // In RPC we don't care the difference as we ser/de with just the
    // plain TensorType. This is not a issue for UserRRef creation either,
    // since Tensor can only get specialized with a previous run of local
    // JIT function, and we shouldn't preserve the specialized SubTensorType
    // information on other workers because it's only information only.
    if (type->isSubtypeOf(TensorType::get())) {
      TORCH_INTERNAL_ASSERT(
          ownerRRef->type()->isSubtypeOf(TensorType::get()),
          "Expect OwnerRRef to be a sub-type of TensorType, but got ",
          ownerRRef->type()->repr_str());
    } else {
      TORCH_INTERNAL_ASSERT(
          *ownerRRef->type() == *type,
          "OwnerRRef type is ",
          ownerRRef->type()->repr_str(),
          ", expected type is ",
          type->repr_str());
    }
    return ownerRRef;
  }
}

c10::intrusive_ptr<OwnerRRef> RRefContext::createOwnerRRef(
    const TypePtr& type) {
  // Don't add this OnwerRRef to the owners_ map yet, otherwise
  // it will never be removed from there. Instead, only add it to the
  // map in prepareChildFork, in case this local RRef is being passed
  // to another worker.
  return c10::make_intrusive<OwnerRRef>(
      getWorkerId(), genGloballyUniqueId(), type, agent_->getDevices());
}

c10::intrusive_ptr<JitFuture> RRefContext::getOwnerRRef(
    const RRefId& rrefId,
    bool forceCreated) {
  std::unique_lock<std::mutex> lock(mutex_);
  const auto iter = owners_.find(rrefId);
  if (iter == owners_.end()) {
    if (forceCreated) {
      TORCH_INTERNAL_ASSERT(
          false,
          c10::str("Expected OwnerRRef with id ", rrefId, " to be created."));
    }
    // Scenario (1) RRef is used before it is created
    const auto pendingOwnerIter = pendingOwners_.find(rrefId);
    if (pendingOwnerIter == pendingOwners_.end()) {
      // Note: The type passed into RRefType::create() does not matter here, as
      // the future is marked as completed with the RRef of the specific type
      // in getOrCreateOwnerRRef().
      // We need to set devices here, even if they won't be used by the value
      // (an RRef object doesn't contain any tensors, it just provides means to
      // retrieve them) because we need them to be propagated/ to child futures.
      // This is silly and we should find a way to avoid this.
      auto futureOwner = c10::make_intrusive<JitFuture>(
          RRefType::create(c10::AnyType::get()), agent_->getDevices());
      pendingOwners_[rrefId] = futureOwner;
      return futureOwner;
    } else {
      return pendingOwnerIter->second;
    }
  } else {
    // Scenario (2) retrieving an existing RRef
    // Marks IValue Future as completed with the RRef IValue.
    auto owner = iter->second;
    auto rrefPtr = fromOwnerRRef(owner);

    // We need to set devices here, even if they won't be used by the value (an
    // RRef object doesn't contain any tensors, it just provides means to
    // retrieve them) because we need them to be propagated/ to child futures.
    // This is silly and we should find a way to avoid this.
    auto futureOwner = c10::make_intrusive<JitFuture>(
        RRefType::create(owner->type()), agent_->getDevices());
    futureOwner->markCompleted(IValue(rrefPtr));
    return futureOwner;
  }
}

RRefForkData RRefContext::prepareChildFork(
    const c10::intrusive_ptr<RRef>& rref) {
  // If we know that rref creation on the owner has timed out, raise it to the
  // user here, otherwise continue with pickling.

  TORCH_CHECK(
      !rref->getTimedOut(),
      "RRef creation via rpc.remote() timed out, and it "
      "is possible that the RRef on the owner node does not exist.");
  auto rrefForkData = rref->fork();
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
    addForkOfOwner(rrefForkData.rrefId_, rrefForkData.forkId_);
    // ensure that this RRef is in the owners_ list to keep it alive.
    // this is needed for OwnerRRefs that were created locally.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      owners_[rref->rrefId()] = rref;
    }
  } else {
    // Note [Useful Phantom Fork ID for User to Owner Call]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // If the callee of dist.remote or dist.rpc is the owner of this RRef, the
    // callee will not create a fork using this rrefForkData.forkId_, because
    // the owner will only keep one `OwnerRRef` instance and will not create any
    // `UserRRef` instances. However, this rrefForkData.forkId_ is still
    // necessary, as the caller user needs to keep this `UserRRef` alive until
    // it gets the ACK from the callee owner. Otherwise, the delete message
    // could arrive at the owner before this dist.rpc or dist.remote call, which
    // could potentially trigger the `OwnerRRef` to be deleted before running
    // the user code.
    addPendingChild(rrefForkData.forkId_, rref);
  }
  return rrefForkData;
}

void RRefContext::notifyOwnerAndParentOfFork(
    const ForkId& forkId,
    worker_id_t parent,
    const c10::intrusive_ptr<RRef>& rref) {
  // Fork is shared from owner.
  if (parent == rref->owner()) {
    if (parent == agent_->getWorkerInfo().id_) {
      // Owner sending RRef to self, remove the forkId as it was added during
      // pickling
      auto deletedRRef = delForkOfOwner(rref->rrefId(), forkId);
      if (deletedRRef) {
        TORCH_INTERNAL_ASSERT(
            deletedRRef->rrefId() == rref->rrefId(),
            "Deleting a fork of ",
            rref->rrefId(),
            " triggered deleting the OwnerRRef of ",
            deletedRRef->rrefId());
        // NB: not necessary to reset deletedRRef as rref is another shared_ptr
        // instance pointing to the same OwnerRRef.
      }
    } else {
      // If the parent is the owner, this fork has already been added into the
      // forks_ map when the owner sends the message to the callee user.
      // Hence, it is not necessary to send another RREF_CHILD_ACCEPT or
      // RREF_FORK_REQUEST back to the owner. See Note [Early Fork
      // Registration].
      std::lock_guard<std::mutex> lock(mutex_);
      addConfirmedUser(forkId, rref);
    }
    return;
  }

  // Fork is shared from user.
  if (rref->isOwner()) {
    // See Note [Useful Phantom Fork ID for User to Owner Call]
    // In this case, the owner is the caller, and it does not add the fork id
    // into forks_. Because, there will be no real `UserRRef` associated
    // with this fork ID.
    ++numPendingFutures_;
    auto jitFuture = agent_->sendWithRetries(
        agent_->getWorkerInfo(parent), RRefChildAccept(forkId).toMessage());
    jitFuture->addCallback([this](JitFuture& future) {
      handleException(future);
      --numPendingFutures_;
    });
  } else {
    ++numPendingFutures_;
    auto jitFuture = agent_->sendWithRetries(
        agent_->getWorkerInfo(rref->owner()),
        RRefForkRequest(rref->rrefId(), forkId).toMessage());

    addPendingUser(forkId, rref);

    jitFuture->addCallback([this, forkId, parent](JitFuture& future) {
      handleException(future);
      this->finishForkRequest(forkId, parent);
      // Decrease after calling finishForkRequest because, as that creates a new
      // future, it might otherwise cause the count to briefly go to zero.
      --numPendingFutures_;
    });
  }
}

void RRefContext::addPendingChild(
    const ForkId& forkId,
    const c10::intrusive_ptr<RRef>& rref) {
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
  c10::intrusive_ptr<RRef> deletedUser;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = pendingChildren_.find(forkId);
    // We first check whether the child exists in pendingChildren_. It's
    // possible the child may have been removed by a previous send attempt, and
    // this check (as opposed to an assertion here) ensures that messages that
    // trigger this function are idempotent.
    if (iter != pendingChildren_.end()) {
      // Since this UserRRef is removed from the map,
      // the refcount of this UserRRef could reach to 0,
      // so the "destructor", `release_resources()`, might be called,
      // in which the lock is acquired again.
      // So it must be destructed with the lock released.
      // Meet this constraint by creating a temporary pointer to increase the
      // refcount, extending its lifetime until lock released.
      deletedUser = iter->second; // Increase refcount.
      pendingChildren_.erase(iter); // Decrease refcount.
    } else {
      LOG(INFO) << "Ignoring duplicate request to delete child UserRRef with "
                << "ForkId = " << forkId;
    }
  }
  deleteAllUsersCV_.notify_all();
  // The refcount of this UserRRef could reach to 0,
  // so the "destructor", release_resources(), might be called,
  // in which the lock is acquired again,
  // so must destruct it with the lock released.
  deletedUser.reset(); // Decrease refcount.
}

void RRefContext::addPendingUser(
    const ForkId& forkId,
    const c10::intrusive_ptr<RRef>& rref) {
  TORCH_INTERNAL_ASSERT(
      !rref->isOwner(), "Attempt to add an OwnerRRef as a pending User.");

  auto state = std::make_shared<PendingUserState>(rref);
  if (recording_) {
    // adding and waiting for pending users are guaranteed to be called from the
    // same thread, but deleting pending users will be called from another
    // thread. As the delPendingUser will not be able to access the same
    // thread_local variable, we cannot address this problem by making
    // pendingUsers_ thread_local. Instead, pendingUsers_ and userTable_ share
    // the same PendingUserState shared_ptr.
    userTable_.push_back(state);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_INTERNAL_ASSERT(
      pendingUsers_.find(forkId) == pendingUsers_.end(),
      "Inconsistent states: attempt to add the same UserRRef twice.");

  pendingUsers_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(forkId),
      std::forward_as_tuple(state));
}

void RRefContext::delPendingUser(const ForkId& forkId) {
  std::shared_ptr<PendingUserState> deletedState = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = pendingUsers_.find(forkId);
    TORCH_INTERNAL_ASSERT(
        iter != pendingUsers_.end(),
        "Inconsistent states: attempt to delete a non-exist UserRRef.");

    // There are two reasons for keeping the deleted PendingUserState alive
    // until exiting the critical section.
    // (1) Since this UserRRef is removed from the map, the refcount of this
    //     UserRRef could reach to 0. So the resource destructor
    //     (`release_resources()`) might be called, in which the lock is
    //     acquired again. Hence, it must be destructed with the lock released.
    //     To meet this constraint, we intentionally create a temporary pointer
    //     to increase the refcount of the deleted PendingUserState, extending
    //     its lifetime until lock released.
    // (2) Since #34497, a user function only runs after all RRefs in the
    //     arguments are confirmed by their owners, which is done by adding the
    //     RPC processing logic as a callback to the UserRRef ready future. So,
    //     calling `confirm` on the PendingUserState could trigger pending user
    //     functions, which might in turn acquire the lock in RRefContext.
    //     Hence, we must release the lock to prevent deadlock.
    // NB: Another option is to use reentrant lock. However, it is better for
    // the developers to fully understand the locking behavior instead of
    // hiding the subtle logic using a reentrant lock.
    deletedState = iter->second; // Increase refcount

    addConfirmedUser(forkId, iter->second->rref_);
    pendingUsers_.erase(iter); // Decrease refcount.
  }
  deletedState->confirm();
  deleteAllUsersCV_.notify_all();
  deletedState.reset(); // Decrease refcount.
}

void RRefContext::addConfirmedUser(
    const ForkId& forkId,
    const c10::intrusive_ptr<RRef>& rref) {
  // Notice, caller need to hold the mutex for confirmedUsers_.
  // std::lock_guard<std::mutex> lock(mutex_);
  confirmedUsers_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(forkId),
      std::forward_as_tuple(rref));
}

c10::intrusive_ptr<RRef> RRefContext::getPendingUser(const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pendingUsers_.find(forkId);
  if (it == pendingUsers_.end()) {
    TORCH_INTERNAL_ASSERT(
        false, "Pending user with forkId ", forkId, " not found");
  }
  return it->second->rref_;
}

void RRefContext::recordThreadLocalPendingRRefs() {
  TORCH_INTERNAL_ASSERT(
      userTable_.empty(),
      "User RRef Table should be empty when start recording");
  recording_ = true;
}

c10::intrusive_ptr<JitFuture> RRefContext::waitForThreadLocalPendingRRefs() {
  // We need to set devices here, even if they won't be used by the value (it's
  // a bool, it doesn't contain tensors!) because we need them to be propagated
  // to child futures. This is silly and we should find a way to avoid this.
  auto jitFuturePtr =
      c10::make_intrusive<JitFuture>(BoolType::get(), agent_->getDevices());
  if (userTable_.empty()) {
    jitFuturePtr->markCompleted(true);
  } else {
    auto remainingRRefs =
        std::make_shared<std::atomic<uint64_t>>(userTable_.size());
    for (auto& state : userTable_) {
      state->confirmationFuture_->addCallback(
          [jitFuturePtr, remainingRRefs](JitFuture& /* unused */) {
            auto localCount = remainingRRefs->fetch_sub(1);
            if (localCount == 1) {
              jitFuturePtr->markCompleted(true);
            }
          });
    }
    userTable_.clear();
  }
  recording_ = false;
  return jitFuturePtr;
}

void RRefContext::clearRecordedPendingRRefsOnError() {
  userTable_.clear();
  recording_ = false;
}

void RRefContext::finishForkRequest(const ForkId& forkId, worker_id_t parent) {
  delPendingUser(forkId);
  ++numPendingFutures_;
  auto jitFuture = agent_->sendWithRetries(
      agent_->getWorkerInfo(parent), RRefChildAccept(forkId).toMessage());

  jitFuture->addCallback([this](JitFuture& future) {
    handleException(future);
    --numPendingFutures_;
  });
}

void RRefContext::addSelfAsFork(c10::intrusive_ptr<OwnerRRef>& rref) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto& rrefId = rref->rrefId();
  owners_[rrefId] = rref;
  auto& rrefForks = forks_[rrefId];
  TORCH_INTERNAL_ASSERT(
      rrefForks.find(rrefId) == rrefForks.end(),
      "Attempt to add self as fork twice ",
      rrefId);
  rrefForks.insert(rrefId);
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

void RRefContext::addForkOfOwnerIfNotPresent(
    const RRefId& rrefId,
    const ForkId& forkId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& rrefForks = forks_[rrefId];
  // We first check whether the child exists in rrefForks. It's possible
  // the child may have been added by a previous send attempt, and this check
  // (as opposed to an assertion here) ensures that messages that trigger this
  // function are idempotent.
  if (rrefForks.find(forkId) == rrefForks.end()) {
    rrefForks.insert(forkId);
  } else {
    LOG(INFO) << "Ignoring duplicate request to add Fork of OwnerRRef with "
              << "RRefId = " << rrefId << ", ForkId = " << forkId;
  }
}

c10::intrusive_ptr<RRef> RRefContext::delForkOfOwner(
    const RRefId& rrefId,
    const ForkId& forkId) {
  c10::intrusive_ptr<RRef> deletedRRef;
  bool ownerReduced = false;
  // There were previously multiple TORCH_CHECKs in this function that checked
  // whether the passed in fork was known by the user and whether the fork had
  // already been deleted. These assertions are now replaced with nested if
  // statements to ensure this function is idempotent. This makes it safe to
  // retry RRefUserDelete messages.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto rrefIter = forks_.find(rrefId);
    if (rrefIter != forks_.end()) {
      auto& rrefForks = rrefIter->second;
      auto forkIter = rrefForks.find(forkId);
      if (forkIter != rrefForks.end()) {
        rrefForks.erase(forkId);
      } else {
        LOG(INFO)
            << "Could not find UserRRef instance, "
            << "RRefId = " << rrefId << ", ForkId = " << forkId
            << ", likely because it was deleted by a previously retried message";
      }
      if (rrefForks.empty()) {
        auto ownerIter = owners_.find(rrefId);
        if (ownerIter != owners_.end()) {
          deletedRRef = ownerIter->second;
          owners_.erase(ownerIter);
          ownerReduced = true;
        }
        forks_.erase(rrefIter);
      }
    } else {
      LOG(INFO)
          << "Could not find OwnerRRef with RRefId = " << rrefId
          << ", likely because it was deleted by a previously retried message";
    }
  }
  if (ownerReduced) {
    deleteAllUsersCV_.notify_all();
  }
  return deletedRRef;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
