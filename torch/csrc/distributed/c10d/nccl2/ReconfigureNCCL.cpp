// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Reconfigure (fault tolerance) implementation for the nccl2 backend. The rest
// of the backend lives in ProcessGroupNCCL*.cpp; this file holds the handle
// encoding and the reconfigure() entry point so the membership-change logic is
// isolated from the collective implementations. The handle format and the
// rank-assignment contract (ordered handles assign ranks by position) match
// ProcessGroupGloo's reconfigure; the communicator teardown/bootstrap steps
// are a port of torchcomms' TorchCommNCCLReconfigure fresh-init path. The
// torchcomms quorum shrink/grow fast path is intentionally not ported: it
// assigns ranks by NCCL's shrink ordering, which conflicts with c10d's
// ordered-handle rank assignment.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

#include <algorithm>
#include <cstring>
#include <exception>
#include <unordered_set>
#include <variant>

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/NCCLBootstrap.hpp>

namespace c10d::nccl2 {

namespace {

std::string getStoreAddress(const c10::intrusive_ptr<::c10d::Store>& store) {
  auto* tcpStore = dynamic_cast<::c10d::TCPStore*>(store.get());
  if (tcpStore == nullptr) {
    auto* prefixStore = dynamic_cast<::c10d::PrefixStore*>(store.get());
    if (prefixStore != nullptr) {
      tcpStore = dynamic_cast<::c10d::TCPStore*>(
          prefixStore->getUnderlyingNonPrefixStore().get());
    }
  }
  if (tcpStore == nullptr) {
    return "";
  }
  return c10::str(tcpStore->getHost(), ":", tcpStore->getPort());
}

struct NCCLReconfigureHandle {
  int rank;
  int64_t uuid;
  std::string storeAddress;
};

NCCLReconfigureHandle parseNCCLReconfigureHandle(
    const ::c10d::ReconfigureHandle& handle) {
  auto first = handle.find(':');
  TORCH_CHECK(
      first != std::string::npos &&
          handle.substr(0, first) == ProcessGroupNCCL::kBackendName,
      "Invalid nccl2 reconfigure handle: ",
      handle);
  auto second = handle.find(':', first + 1);
  TORCH_CHECK(
      second != std::string::npos,
      "Invalid nccl2 reconfigure handle: ",
      handle);
  auto third = handle.find(':', second + 1);
  TORCH_CHECK(
      third != std::string::npos, "Invalid nccl2 reconfigure handle: ", handle);
  return {
      .rank = std::stoi(handle.substr(first + 1, second - first - 1)),
      .uuid = std::stoll(handle.substr(second + 1, third - second - 1)),
      .storeAddress = handle.substr(third + 1)};
}

std::vector<::c10d::ReconfigureHandle> getOrderedReconfigureHandles(
    const ::c10d::ReconfigureOptions& opts) {
  std::vector<::c10d::ReconfigureHandle> handles;
  std::visit(
      [&](const auto& inputHandles) {
        handles.assign(inputHandles.begin(), inputHandles.end());
      },
      opts.handles);
  if (std::holds_alternative<std::unordered_set<::c10d::ReconfigureHandle>>(
          opts.handles)) {
    std::ranges::sort(handles);
  }
  TORCH_CHECK(!handles.empty(), "Reconfigure requires at least one handle");
  std::unordered_set<::c10d::ReconfigureHandle> uniqueHandles(
      handles.begin(), handles.end());
  TORCH_CHECK(
      uniqueHandles.size() == handles.size(),
      "Reconfigure handles must be unique");
  for (const auto& handle : handles) {
    parseNCCLReconfigureHandle(handle);
  }
  return handles;
}

c10::intrusive_ptr<::c10d::Work> makeCompletedWork() {
  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), std::vector<at::Device>{});
  future->markCompleted(c10::IValue(std::vector<at::Tensor>()));
  return ::c10d::Work::create_from_future(future);
}

} // namespace

::c10d::ReconfigureHandle ProcessGroupNCCL::get_reconfigure_handle() const {
  return c10::str(
      kBackendName,
      ":",
      rank_,
      ":",
      reconfigure_uuid_,
      ":",
      getStoreAddress(store_));
}

c10::intrusive_ptr<::c10d::Work> ProcessGroupNCCL::reconfigure(
    const ::c10d::ReconfigureOptions& opts) {
  std::unique_lock reconfigureLock(reconfigure_mutex_);
  TORCH_CHECK(
      !reconfiguring_.load(std::memory_order_acquire),
      "ProcessGroupNCCL reconfigure is already in progress");
  TORCH_CHECK(
      init_state_ != InitializationState::FINALIZED,
      "ProcessGroupNCCL has been finalized");
  auto handles = getOrderedReconfigureHandles(opts);
  auto localHandle = get_reconfigure_handle();
  auto localIt = std::ranges::find(handles, localHandle);
  TORCH_CHECK(
      localIt != handles.end(),
      "Local nccl2 reconfigure handle is not part of the new communicator");
  auto newRank = static_cast<int>(std::distance(handles.begin(), localIt));
  auto newSize = static_cast<int>(handles.size());
  auto timeout = opts.timeout.value_or(options_c10d_->timeout);

  TC_LOG(INFO, this) << "ProcessGroupNCCL reconfigure starting: uuid="
                     << opts.uuid << " new_rank=" << newRank
                     << " new_size=" << newSize;

  auto prefixedStore = c10::make_intrusive<::c10d::PrefixStore>(
      c10::str("nccl2_reconfigure/", opts.uuid), store_);
  if (!nccl_api_) {
    nccl_api_ = std::make_shared<DefaultNcclApi>();
  }

  // The uuid namespaces this reconfigure's rendezvous keys. Rank 0 claims it
  // before publishing a unique ID so a reused uuid cannot overwrite a live
  // rendezvous.
  if (newRank == 0) {
    auto claimedBy = prefixedStore->compareSet("claimed", "", localHandle);
    TORCH_CHECK(
        claimedBy == localHandle,
        "nccl2 reconfigure uuid ",
        opts.uuid,
        " was already used; each reconfigure() requires a unique uuid");
  }

  // Exchange the ncclUniqueId before tearing down the current communicator.
  // Including the current rank-0 handle prevents nonzero ranks from consuming
  // a stale ID when rank 0 rejects a reused uuid; they instead time out here.
  const auto uniqueIdKey = c10::str("unique_id/", handles.front());
  ncclUniqueId uniqueId{};
  if (newRank == 0) {
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->getUniqueId(&uniqueId),
        "NCCL getUniqueId failed during reconfigure");
    std::vector<uint8_t> vec(
        reinterpret_cast<uint8_t*>(&uniqueId),
        reinterpret_cast<uint8_t*>(&uniqueId) + sizeof(uniqueId));
    prefixedStore->set(uniqueIdKey, vec);
  } else {
    prefixedStore->wait({uniqueIdKey}, timeout);
    auto vec = prefixedStore->get(uniqueIdKey);
    TORCH_CHECK(
        vec.size() == sizeof(ncclUniqueId),
        "Invalid NCCL unique ID size during reconfigure");
    std::memcpy(&uniqueId, vec.data(), sizeof(ncclUniqueId));
  }

  // Close admission before waiting for already admitted collectives to finish
  // enqueueing. Those operations hold a shared admission lock from createWork
  // through queue insertion; this exclusive lock therefore makes the following
  // queue snapshot complete for all work launched before the transition.
  reconfiguring_.store(true, std::memory_order_release);
  reconfigure_epoch_.fetch_add(1, std::memory_order_acq_rel);
  std::unique_lock<std::shared_mutex> admissionLock(
      collective_admission_mutex_);

  // This generation mutex is the success/cancel linearization point. A work
  // completion that owns it first may commit success; otherwise this marker
  // makes every subsequent event-query success ineligible.
  {
    std::lock_guard<std::mutex> generationLock(work_generation_state_->mutex);
    work_generation_state_->invalidated = true;
  }

  // Tear down the previous communicator generation. Stop the watchdog before
  // taking ownership of revoke; if it already started a revoke, revoked_ tells
  // us to wait for that operation rather than issuing a second revoke.
  if (init_state_ == InitializationState::INITIALIZED) {
    // Queue collection and cancellation resolve Futures and can synchronously
    // run user callbacks. The generation is already closed, so temporarily
    // release both transition locks while those callbacks execute; a nested
    // reconfigure will observe reconfiguring_ and fail instead of deadlocking.
    reconfigureLock.unlock();
    admissionLock.unlock();
    stopWatchdog();
    workq_.garbageCollect();
    failPendingGeneration(reconfigure_uuid_);
    reconfigureLock.lock();
    admissionLock.lock();

    if (nccl_comm_) {
      const auto oldComm = nccl_comm_;
      bool alreadyRevoked = revoked_.load(std::memory_order_acquire);
      if (!alreadyRevoked) {
        // Abort hooks are user callbacks. Run them outside transition locks,
        // then re-check ownership because a hook may itself abort the group.
        reconfigureLock.unlock();
        admissionLock.unlock();
        runAbortHooks();
        reconfigureLock.lock();
        admissionLock.lock();
      }
      alreadyRevoked = revoked_.exchange(true);
      if (!alreadyRevoked) {
        detachMemoryHook();
        retireComm();
      }

      // Revoke is nonblocking on a nonblocking communicator. If the watchdog
      // already initiated it, query the current async state; otherwise issue
      // the revoke now. Revoke errors remain best-effort because the following
      // abort is the final resource cleanup step.
      try {
        ncclResult_t revokeStatus = ncclSuccess;
        if (alreadyRevoked) {
          revokeStatus = ncclInProgress;
          const auto queryStatus =
              nccl_api_->commGetAsyncError(oldComm, &revokeStatus);
          if (queryStatus != ncclSuccess) {
            throw NCCLException(
                *nccl_api_,
                "NCCL async error query failed while waiting for revoke "
                "during reconfigure",
                queryStatus,
                oldComm);
          }
        } else {
          revokeStatus = nccl_api_->commRevoke(oldComm);
        }
        waitForNcclCompletion(
            *nccl_api_,
            oldComm,
            revokeStatus,
            timeout,
            "NCCL commRevoke failed during reconfigure");
      } catch (const std::exception& e) {
        LOG(ERROR) << e.what();
      }

      waitForNcclCompletion(
          *nccl_api_,
          oldComm,
          nccl_api_->commAbort(oldComm),
          timeout,
          "NCCL commAbort failed during reconfigure");
      nccl_comm_ = nullptr;
      init_state_ = InitializationState::UNINITIALIZED;
    }
    // Keep queued tensor shelves alive until revoke/abort has settled.
    // finalize() may notify successful Works that were not popped by the
    // earlier queue poll, so keep user completion hooks outside transition
    // locks as well.
    reconfigureLock.unlock();
    admissionLock.unlock();
    workq_.finalize();
    reconfigureLock.lock();
    admissionLock.lock();
  }
  init_state_ = InitializationState::UNINITIALIZED;

  comm_state_ = CommState::NORMAL;
  shutdown_ = false;
  revoked_ = false;

  // Resolve the device on the first reconfigure: prefer the bound device,
  // else the caller's current CUDA device. The bootstrap's rank-based default
  // would collide when disjoint single-rank groups reconfigure concurrently
  // (every group's rank 0 would land on cuda:0).
  at::Device device = device_;
  if (device.index() == -1) {
    if (getBoundDeviceId().has_value()) {
      device = getBoundDeviceId().value();
    } else {
      device = at::Device(at::kCUDA, at::cuda::current_device());
    }
  }

  rank_ = newRank;
  size_ = newSize;
  device_ = device;

  c10::cuda::CUDAGuard gpuGuard(device_);

  ncclConfig_t config = options_c10d_->config;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  config.commName = name_.c_str();
#endif
  populateNcclConfigFromHints(config, opts.hints, name_);
  // ReconfigureOptions::timeout must bound initialization even when the
  // process-group config normally uses blocking NCCL calls.
  config.blocking = 0;

  ncclComm_t new_comm = nullptr;
  try {
    auto init_status = nccl_api_->commInitRankConfig(
        &new_comm, newSize, uniqueId, newRank, &config);
    TORCH_CHECK(
        new_comm,
        "NCCL commInitRankConfig failed during reconfigure: ",
        nccl_api_->getErrorString(init_status));
    waitForNcclCompletion(
        *nccl_api_,
        new_comm,
        init_status,
        timeout,
        "NCCL commInitRankConfig failed during reconfigure");
  } catch (...) {
    if (new_comm != nullptr) {
      try {
        waitForNcclCompletion(
            *nccl_api_,
            new_comm,
            nccl_api_->commAbort(new_comm),
            timeout,
            "NCCL commAbort failed after reconfigure initialization failure");
      } catch (const std::exception& e) {
        LOG(ERROR) << e.what();
      }
    }
    comm_state_ = CommState::ERROR;
    nccl_comm_ = nullptr;
    init_state_ = InitializationState::UNINITIALIZED;
    throw;
  }
  nccl_comm_ = new_comm;
  try {
    initNcclResources();
  } catch (...) {
    const auto initException = std::current_exception();
    try {
      stopWatchdog();
      detachMemoryHook();
      retireComm();
      waitForNcclCompletion(
          *nccl_api_,
          new_comm,
          nccl_api_->commAbort(new_comm),
          timeout,
          "NCCL commAbort failed after resource initialization failure");
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to clean up replacement NCCL communicator: "
                 << e.what();
    }
    comm_state_ = CommState::ERROR;
    nccl_comm_ = nullptr;
    init_state_ = InitializationState::UNINITIALIZED;
    std::rethrow_exception(initException);
  }
  reconfigure_uuid_ = opts.uuid;
  work_generation_state_ = std::make_shared<WorkGenerationState>();
  init_state_ = InitializationState::INITIALIZED;
  reconfiguring_.store(false, std::memory_order_release);

  TC_LOG(INFO, this) << "ProcessGroupNCCL reconfigure completed for rank: "
                     << rank_;

  return makeCompletedWork();
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
