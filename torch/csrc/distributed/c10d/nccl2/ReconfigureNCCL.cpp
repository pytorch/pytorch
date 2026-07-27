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

  // The uuid namespaces this reconfigure's rendezvous keys; reusing it would
  // read stale rendezvous state. New rank 0 atomically claims the uuid: the
  // compareSet writes our handle only while "claimed" is unset, so a reused
  // uuid returns the prior claimant's handle, which differs from ours.
  if (newRank == 0) {
    auto claimedBy = prefixedStore->compareSet("claimed", "", localHandle);
    TORCH_CHECK(
        claimedBy == localHandle,
        "nccl2 reconfigure uuid ",
        opts.uuid,
        " was already used; each reconfigure() requires a unique uuid");
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
      NCCL_CHECK_IGNORE(
          nccl_api_,
          nccl_api_->commAbort(nccl_comm_),
          "NCCL commAbort failed during reconfigure");
      nccl_comm_ = nullptr;
    }
  }

  if (!nccl_api_) {
    nccl_api_ = std::make_shared<DefaultNcclApi>();
  }

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

  // Exchange the ncclUniqueId through the uuid-namespaced store with a fixed
  // key. NCCLBootstrap is not reused here: its store keys embed a
  // process-wide static counter, which diverges across ranks when disjoint
  // groups reconfigure a different number of times (e.g. a late-joining rank).
  static const char* kUniqueIdKey = "unique_id";
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
    prefixedStore->set(kUniqueIdKey, vec);
  } else {
    prefixedStore->wait({kUniqueIdKey}, timeout);
    auto vec = prefixedStore->get(kUniqueIdKey);
    TORCH_CHECK(
        vec.size() == sizeof(ncclUniqueId),
        "Invalid NCCL unique ID size during reconfigure");
    std::memcpy(&uniqueId, vec.data(), sizeof(ncclUniqueId));
  }

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  config.commName = name_.c_str();
#endif
  populateNcclConfigFromHints(config, options_c10d_->hints, name_);

  ncclComm_t new_comm = nullptr;
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commInitRankConfig(
          &new_comm, newSize, uniqueId, newRank, &config),
      "NCCL commInitRankConfig failed during reconfigure");
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
