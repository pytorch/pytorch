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

#include <ATen/cuda/CUDAContext.h>
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
  std::lock_guard reconfigureLock(reconfigure_mutex_);
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

  // Tear down the previous communicator generation: revoke in-flight work,
  // stop the watchdog, drain the work queue, and abort the comm. Port of the
  // pre-reconfigure cleanup in torchcomms' TorchCommNCCL::reconfigure.
  if (init_state_ == InitializationState::INITIALIZED) {
    auto workStatus = workq_.garbageCollect();
    if (nccl_comm_ &&
        (workStatus == WorkNCCL::WorkStatus::NOT_STARTED ||
         workStatus == WorkNCCL::WorkStatus::INPROGRESS)) {
      NCCL_CHECK_IGNORE(
          nccl_api_,
          nccl_api_->commRevoke(nccl_comm_),
          "NCCL commRevoke failed during reconfigure");
    }

    detachMemoryHook();
    retireComm();

    if (timeout_thread_.joinable()) {
      shutdown_ = true;
      {
        std::lock_guard<std::mutex> lock(timeout_mutex_);
        timeout_cv_.notify_all();
      }
      timeout_thread_.join();
    }

    workq_.finalize();

    if (nccl_comm_) {
      auto oldComm = std::exchange(nccl_comm_, nullptr);
      init_state_ = InitializationState::UNINITIALIZED;
      waitForNcclCompletion(
          *nccl_api_,
          oldComm,
          nccl_api_->commAbort(oldComm),
          timeout,
          "NCCL commAbort failed during reconfigure");
    }
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

  initNcclResources();
  init_state_ = InitializationState::INITIALIZED;
  reconfigure_uuid_ = opts.uuid;

  TC_LOG(INFO, this) << "ProcessGroupNCCL reconfigure completed for rank: "
                     << rank_;

  return makeCompletedWork();
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
