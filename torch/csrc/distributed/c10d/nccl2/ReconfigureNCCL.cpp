// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Reconfigure (fault tolerance) implementation for the nccl2 backend. The rest
// of the backend lives in ProcessGroupNCCL*.cpp; this file holds the handle
// encoding and the reconfigure() entry point so the membership-change logic is
// isolated from the collective implementations. The handle format and the
// rank-assignment contract (ordered handles assign ranks by position) match
// ProcessGroupGloo's reconfigure; the communicator teardown/bootstrap steps
// are a port of torchcomms' TorchCommNCCLReconfigure. The torchcomms quorum
// shrink/grow fast path is opt-in (TORCH_NCCL2_RECONFIGURE_SHRINK_GROW=1, set
// consistently on every rank) and only taken when NCCL's shrink/grow rank
// order matches c10d's ordered-handle rank assignment; otherwise every rank
// falls back to a fresh commInitRankConfig.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

#include <algorithm>
#include <cstring>
#include <exception>
#include <map>
#include <thread>
#include <unordered_set>
#include <variant>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/env.h>
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
  auto fourth = handle.find(':', third + 1);
  TORCH_CHECK(
      fourth != std::string::npos,
      "Invalid nccl2 reconfigure handle: ",
      handle);
  return {
      .rank = std::stoi(handle.substr(first + 1, second - first - 1)),
      .uuid = std::stoll(handle.substr(second + 1, third - second - 1)),
      .storeAddress = handle.substr(fourth + 1)};
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

struct ReconfigureQuorum {
  int64_t uuid = -1;
  // Number of leading handles that keep their communicator; 0 = fresh init.
  int size = 0;
};

// Picks the largest set of ranks sharing a previous communicator (ties go to
// the larger uuid). The decision depends only on `handles`, so every rank
// picks the same path. Shrink keeps survivors in old-rank order and grow
// appends new ranks, so the fast path requires the quorum to come first in
// ascending old-rank order.
ReconfigureQuorum findReconfigureQuorum(
    const std::vector<::c10d::ReconfigureHandle>& handles) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  if (!c10::utils::check_env("TORCH_NCCL2_RECONFIGURE_SHRINK_GROW")
           .value_or(false)) {
    return {};
  }
  std::map<int64_t, int> countByUuid;
  for (const auto& handle : handles) {
    auto uuid = parseNCCLReconfigureHandle(handle).uuid;
    if (uuid >= 0) {
      ++countByUuid[uuid];
    }
  }
  ReconfigureQuorum quorum;
  for (const auto& [uuid, count] : countByUuid) {
    if (count >= quorum.size) {
      quorum = {.uuid = uuid, .size = count};
    }
  }
  // A single-rank communicator has no bootstrap network to grow from.
  if (quorum.size < 2) {
    return {};
  }
  int previousRank = -1;
  for (int i = 0; i < quorum.size; ++i) {
    auto info = parseNCCLReconfigureHandle(handles[i]);
    if (info.uuid != quorum.uuid || info.rank <= previousRank) {
      return {};
    }
    previousRank = info.rank;
  }
  return quorum;
#else
  std::ignore = handles;
  return {};
#endif
}

void abortCommIgnoringErrors(
    NcclApi& api,
    ncclComm_t comm,
    std::chrono::milliseconds timeout,
    std::string_view operation) {
  try {
    waitForNcclCompletion(api, comm, api.commAbort(comm), timeout, operation);
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
}

std::vector<uint8_t> uniqueIdToBytes(const ncclUniqueId& uniqueId) {
  return {
      reinterpret_cast<const uint8_t*>(&uniqueId),
      reinterpret_cast<const uint8_t*>(&uniqueId) + sizeof(uniqueId)};
}

ncclUniqueId waitForUniqueId(
    ::c10d::Store& store,
    const std::string& key,
    std::chrono::milliseconds timeout) {
  store.wait({key}, timeout);
  auto vec = store.get(key);
  TORCH_CHECK(
      vec.size() == sizeof(ncclUniqueId),
      "Invalid NCCL unique ID size during reconfigure");
  ncclUniqueId uniqueId{};
  std::memcpy(&uniqueId, vec.data(), sizeof(ncclUniqueId));
  return uniqueId;
}

// Shrinks `comm` to exclude `excluded`, then grows it to `newSize`. Takes
// ownership of `comm`: it is aborted on both success and failure.
ncclComm_t shrinkAndGrowComm(
    NcclApi& api,
    ncclComm_t comm,
    std::vector<int> excluded,
    int newSize,
    bool publishGrowId,
    ::c10d::Store& store,
    const std::string& growIdKey,
    ncclConfig_t config,
    std::chrono::milliseconds timeout) {
  // A nonblocking commRevoke may still be in progress; NCCL rejects further
  // calls on the comm until it completes. Errors are fine: shrink with
  // NCCL_SHRINK_ABORT accepts a failed comm.
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  ncclResult_t asyncStatus = ncclInProgress;
  while (api.commGetAsyncError(comm, &asyncStatus) == ncclSuccess &&
         asyncStatus == ncclInProgress) {
    if (std::chrono::steady_clock::now() >= deadline) {
      abortCommIgnoringErrors(
          api, comm, timeout, "NCCL commAbort after revoke timeout failed");
      TORCH_CHECK_WITH(
          DistBackendError,
          false,
          "NCCL commRevoke did not complete during reconfigure within ",
          timeout.count(),
          " ms");
    }
    std::this_thread::yield();
  }

  if (!excluded.empty()) {
    ncclComm_t shrunk = nullptr;
    auto status = api.commShrink(
        comm,
        excluded.data(),
        static_cast<int>(excluded.size()),
        &shrunk,
        &config,
        NCCL_SHRINK_ABORT);
    try {
      waitForNcclChildComm(
          api,
          comm,
          &shrunk,
          status,
          true,
          timeout,
          "NCCL commShrink failed during reconfigure");
    } catch (...) {
      // waitForNcclChildComm aborts the parent unless the call itself failed.
      if (status != ncclSuccess && status != ncclInProgress) {
        abortCommIgnoringErrors(
            api, comm, timeout, "NCCL commAbort failed after commShrink");
      }
      throw;
    }
    abortCommIgnoringErrors(
        api, comm, timeout, "NCCL commAbort of pre-shrink comm failed");
    comm = shrunk;
  }

  int size = 0;
  auto status = api.commCount(comm, &size);
  if (status == ncclSuccess && size == newSize) {
    return comm;
  }
  ncclComm_t grown = nullptr;
  try {
    TORCH_CHECK(
        status == ncclSuccess,
        "NCCL commCount failed during reconfigure: ",
        api.getErrorString(status));
    if (publishGrowId) {
      ncclUniqueId uniqueId{};
      status = api.commGetUniqueId(comm, &uniqueId);
      TORCH_CHECK(
          status == ncclSuccess,
          "NCCL commGetUniqueId failed during reconfigure: ",
          api.getErrorString(status));
      store.set(growIdKey, uniqueIdToBytes(uniqueId));
    } else {
      // commGrow blocks without a timeout until rank 0 sends the grow handle
      // from commGetUniqueId, which happens before the store key is set.
      waitForUniqueId(store, growIdKey, timeout);
    }
    status = api.commGrow(comm, newSize, nullptr, -1, &grown, &config);
  } catch (...) {
    abortCommIgnoringErrors(
        api, comm, timeout, "NCCL commAbort failed after commGrow");
    throw;
  }
  try {
    waitForNcclChildComm(
        api,
        comm,
        &grown,
        status,
        true,
        timeout,
        "NCCL commGrow failed during reconfigure");
  } catch (...) {
    if (status != ncclSuccess && status != ncclInProgress) {
      abortCommIgnoringErrors(
          api, comm, timeout, "NCCL commAbort failed after commGrow");
    }
    throw;
  }
  abortCommIgnoringErrors(
      api, comm, timeout, "NCCL commAbort of pre-grow comm failed");
  return grown;
}

// Creates a communicator with commInitRankConfig, or joins an existing one
// with commGrow when `grow` is set.
ncclComm_t initComm(
    NcclApi& api,
    bool grow,
    int newSize,
    int newRank,
    const ncclUniqueId& uniqueId,
    ncclConfig_t config,
    std::chrono::milliseconds timeout) {
  const auto operation = grow
      ? "NCCL commGrow failed during reconfigure"
      : "NCCL commInitRankConfig failed during reconfigure";
  ncclComm_t comm = nullptr;
  auto status = grow
      ? api.commGrow(nullptr, newSize, &uniqueId, newRank, &comm, &config)
      : api.commInitRankConfig(&comm, newSize, uniqueId, newRank, &config);
  TORCH_CHECK(comm, operation, ": ", api.getErrorString(status));
  // A nonblocking commGrow can return ncclSuccess while the joining comm is
  // still initializing, so always poll its async state.
  if (grow && status == ncclSuccess) {
    status = ncclInProgress;
  }
  try {
    waitForNcclCompletion(api, comm, status, timeout, operation);
  } catch (...) {
    abortCommIgnoringErrors(
        api,
        comm,
        timeout,
        "NCCL commAbort failed after reconfigure initialization failure");
    throw;
  }
  return comm;
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
      reconfigure_instance_id_,
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

  auto quorum = findReconfigureQuorum(handles);
  bool inQuorum = newRank < quorum.size;
  std::vector<int> excluded;
  if (inQuorum) {
    TORCH_CHECK(
        nccl_comm_ && init_state_ == InitializationState::INITIALIZED,
        "nccl2 reconfigure handle names communicator ",
        reconfigure_uuid_,
        " but it is not initialized");
    std::unordered_set<int> keptRanks;
    for (int i = 0; i < quorum.size; ++i) {
      keptRanks.insert(parseNCCLReconfigureHandle(handles[i]).rank);
    }
    for (int rank = 0; rank < comm_size_; ++rank) {
      if (!keptRanks.contains(rank)) {
        excluded.push_back(rank);
      }
    }
    TORCH_CHECK(
        comm_size_ - static_cast<int>(excluded.size()) == quorum.size,
        "nccl2 reconfigure quorum has ranks outside communicator ",
        reconfigure_uuid_);
  }
  // Identity reconfigure: the old comm may be revoked, so re-create it. Only
  // quorum members reach this, and a full quorum has no other ranks.
  if (inQuorum && quorum.size == newSize && excluded.empty()) {
    quorum = {};
    inQuorum = false;
  }
  const bool shrinkGrow = quorum.size > 0;

  TC_LOG(INFO, this) << "ProcessGroupNCCL reconfigure starting: uuid="
                     << opts.uuid << " new_rank=" << newRank
                     << " new_size=" << newSize << " path="
                     << (!shrinkGrow    ? "init"
                             : inQuorum ? "shrink_grow"
                                        : "grow_join");

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

  // Including the current rank-0 handle prevents nonzero ranks from consuming
  // a stale ID when rank 0 rejects a reused uuid; they instead time out here.
  const auto uniqueIdKey = c10::str("unique_id/", handles.front());
  ncclUniqueId uniqueId{};
  // Exchange the ncclUniqueId before tearing down the current communicator.
  // The shrink/grow path publishes it later, from the shrunk communicator.
  if (!shrinkGrow) {
    if (newRank == 0) {
      NCCL_CHECK(
          nccl_api_,
          nccl_comm_,
          nccl_api_->getUniqueId(&uniqueId),
          "NCCL getUniqueId failed during reconfigure");
      prefixedStore->set(uniqueIdKey, uniqueIdToBytes(uniqueId));
    } else {
      uniqueId = waitForUniqueId(*prefixedStore, uniqueIdKey, timeout);
    }
  }

  ncclConfig_t config = options_c10d_->config;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  config.commName = name_.c_str();
#endif
  populateNcclConfigFromHints(config, opts.hints, name_);
  // ReconfigureOptions::timeout must bound initialization even when the
  // process-group config normally uses blocking NCCL calls.
  config.blocking = 0;

  // Tear down the previous communicator generation: revoke in-flight work,
  // stop the watchdog, drain the work queue, and abort the comm unless it is
  // shrunk below. Port of the pre-reconfigure cleanup in torchcomms'
  // TorchCommNCCL::reconfigure.
  ncclComm_t oldComm = nullptr;
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

    oldComm = std::exchange(nccl_comm_, nullptr);
    init_state_ = InitializationState::UNINITIALIZED;
    // The handle must not advertise a communicator this rank no longer has.
    reconfigure_uuid_ = -1;
    if (oldComm && !inQuorum) {
      auto comm = std::exchange(oldComm, nullptr);
      waitForNcclCompletion(
          *nccl_api_,
          comm,
          nccl_api_->commAbort(comm),
          timeout,
          "NCCL commAbort failed during reconfigure");
    }
  }
  init_state_ = InitializationState::UNINITIALIZED;

  comm_state_ = CommState::NORMAL;
  shutdown_ = false;
  revoked_ = false;
  {
    std::lock_guard<std::mutex> lock(timeout_message_mutex_);
    timeout_message_.clear();
  }

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

  const auto growIdKey = c10::str("grow_unique_id/", handles.front());
  try {
    if (inQuorum) {
      nccl_comm_ = shrinkAndGrowComm(
          *nccl_api_,
          oldComm,
          std::move(excluded),
          newSize,
          newRank == 0,
          *prefixedStore,
          growIdKey,
          config,
          timeout);
    } else {
      if (shrinkGrow) {
        uniqueId = waitForUniqueId(*prefixedStore, growIdKey, timeout);
      }
      nccl_comm_ = initComm(
          *nccl_api_, shrinkGrow, newSize, newRank, uniqueId, config, timeout);
    }
  } catch (...) {
    comm_state_ = CommState::ERROR;
    nccl_comm_ = nullptr;
    init_state_ = InitializationState::UNINITIALIZED;
    throw;
  }

  initNcclResources();
  init_state_ = InitializationState::INITIALIZED;
  TORCH_CHECK(
      rank_ == newRank && comm_size_ == newSize,
      "nccl2 reconfigure produced rank ",
      rank_,
      " of ",
      comm_size_,
      ", expected ",
      newRank,
      " of ",
      newSize);
  reconfigure_uuid_ = opts.uuid;

  TC_LOG(INFO, this) << "ProcessGroupNCCL reconfigure completed for rank: "
                     << rank_;

  return makeCompletedWork();
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
