// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/env.h>
#include <fmt/core.h>
#include <nccl.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/csrc/distributed/c10d/NCCLCommRegistrationHook.hpp>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/NCCLBootstrap.hpp>
#include <torch/csrc/distributed/c10d/nccl2/TracingGuard.hpp>
#include <torch/csrc/distributed/c10d/nccl2/Utils.hpp>

namespace c10d::nccl2 {

namespace {

void checkSameDtype(const at::Tensor& reference, const at::Tensor& tensor) {
  if (reference.scalar_type() != tensor.scalar_type()) {
    C10_THROW_ERROR(TypeError, "Tensors must have identical type");
  }
}

void checkSameDtype(
    const at::Tensor& reference,
    const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    checkSameDtype(reference, tensor);
  }
}

} // namespace

ncclConfig_t cloneNcclConfig(const ncclConfig_t& config) {
  ncclConfig_t clone = config;
  if (clone.netName != nullptr) {
    clone.netName = strdup(clone.netName);
  }
  return clone;
}

void waitForNcclCompletion(
    NcclApi& nccl_api,
    ncclComm_t comm,
    ncclResult_t status,
    std::chrono::milliseconds timeout,
    std::string_view operation) {
  const auto start = std::chrono::steady_clock::now();
  while (status == ncclInProgress) {
    TORCH_CHECK_WITH(
        DistBackendError,
        std::chrono::steady_clock::now() - start < timeout,
        operation,
        " timed out after ",
        timeout.count(),
        " ms");
    std::this_thread::yield();
    const auto query_status = nccl_api.commGetAsyncError(comm, &status);
    if (query_status != ncclSuccess) {
      throw NCCLException(
          nccl_api,
          fmt::format("{} async error query failed", operation),
          query_status,
          comm);
    }
  }
  if (status != ncclSuccess) {
    throw NCCLException(nccl_api, std::string(operation), status, comm);
  }
}

void waitForNcclChildComm(
    NcclApi& nccl_api,
    ncclComm_t parent_comm,
    ncclComm_t* child_comm,
    ncclResult_t status,
    bool expect_child,
    std::chrono::milliseconds timeout,
    std::string_view operation) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  const auto remaining = [&] {
    const auto now = std::chrono::steady_clock::now();
    TORCH_CHECK_WITH(
        DistBackendError,
        now < deadline,
        operation,
        " timed out after ",
        timeout.count(),
        " ms");
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - now);
  };
  try {
    waitForNcclCompletion(
        nccl_api, parent_comm, status, remaining(), operation);
    if (!expect_child) {
      return;
    }

    while (*child_comm == nullptr) {
      remaining();
      std::this_thread::yield();
    }

    ncclResult_t child_status{};
    auto query_status = nccl_api.commGetAsyncError(*child_comm, &child_status);
    if (query_status != ncclSuccess) {
      throw NCCLException(
          nccl_api,
          fmt::format("{} child async error query failed", operation),
          query_status,
          *child_comm);
    }
    waitForNcclCompletion(
        nccl_api, *child_comm, child_status, remaining(), operation);
  } catch (...) {
    const auto abortComm = [&](ncclComm_t comm, std::string_view description) {
      try {
        waitForNcclCompletion(
            nccl_api, comm, nccl_api.commAbort(comm), timeout, description);
      } catch (const std::exception& error) {
        LOG(ERROR) << error.what();
      }
    };
    if (status == ncclSuccess || status == ncclInProgress) {
      abortComm(parent_comm, "Failed to abort parent NCCL communicator");
    }
    if (*child_comm != nullptr) {
      abortComm(
          std::exchange(*child_comm, nullptr),
          "Failed to abort child NCCL communicator");
    }
    throw;
  }
}

void ProcessGroupNCCL::waitForNcclOperation(
    ncclResult_t status,
    std::chrono::milliseconds timeout,
    std::string_view operation) {
  waitForNcclCompletion(*nccl_api_, nccl_comm_, status, timeout, operation);
}

ncclResult_t NCCLException::getResult() const noexcept {
  return result_;
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
  if (init_state_ == InitializationState::INITIALIZED) {
    TC_LOG(WARNING, this)
        << "ProcessGroupNCCL " << name_
        << " was not finalized before destruction. "
        << "This may indicate a resource leak. Please call finalize() explicitly.";

    // Stop the watchdog so it cannot access this object after destruction. It
    // may be the caller (garbageCollect can pop a work item whose destruction
    // releases the last reference to this comm), which stopWatchdog handles.
    stopWatchdog();

    // Abort the NCCL communicator since we can't do a clean finalization.
    // Open-coded rather than abortNcclComm() so a failure cannot throw out of
    // the destructor (abortNcclComm() throws on a failed or timed-out abort).
    if (nccl_comm_) {
      // Drop our symmetric-memory registration while nccl_comm_ is still valid
      // (it is nulled below, before detachMemoryHook runs).
      retireComm();
      // Best effort to abort the communicator - ignore errors since we're
      // in the destructor
      if (nccl_api_) {
        try {
          waitForNcclCompletion(
              *nccl_api_,
              nccl_comm_,
              nccl_api_->commAbort(nccl_comm_),
              options_c10d_->timeout,
              "NCCL Abort failed during destruction");
        } catch (const std::exception& e) {
          LOG(ERROR) << e.what();
        }
      }
      nccl_comm_ = nullptr;
    }
  }

  // We need to detach the memory hook in case finalize is not called,
  // so that we don't encounter a memory corruption.
  detachMemoryHook();
}

void ProcessGroupNCCL::init(at::Device device) {
  TC_LOG(INFO, this) << "Initializing ProcessGroupNCCL for device: " << device;
  device_ = device;

  if (init_state_ == InitializationState::INITIALIZED) {
    throw std::runtime_error("ProcessGroupNCCL already initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("ProcessGroupNCCL already finalized");
  }

  if (!nccl_api_) {
    nccl_api_ = std::make_unique<DefaultNcclApi>();
  }

  // If deterministic mode is enabled, disable the NVLS algorithm in NCCL
  // (which can lead to non-deterministic reductions). Mirrors the stock
  // ProcessGroupNCCL. NCCL reads NCCL_ALGO when the communicator is created,
  // so this must run before createNcclComm below. If the user already set
  // NCCL_ALGO, leave it untouched.
  // TODO: remove this once NVLS supports deterministic mode.
  if (at::globalContext().deterministicAlgorithms()) {
    if (!c10::utils::get_env("NCCL_ALGO").has_value()) {
      LOG(INFO)
          << "torch deterministic mode is enabled, "
          << "disabling NVLS algorithm in NCCL which can lead to non-deterministic reduction.";
      c10::utils::set_env("NCCL_ALGO", "^NVLS");
    }
  }

  if (device_.index() == -1 || nccl_comm_ == nullptr) {
    auto bootstrap = std::make_unique<NCCLBootstrap>(
        store_,
        device_,
        getRank(),
        getSize(),
        bootstrap_generation_++,
        nccl_api_,
        options_c10d_->timeout);
    device_ = bootstrap->getDevice();

    if (nccl_comm_ == nullptr) {
      nccl_comm_ = bootstrap->createNcclComm(name_, options_c10d_->config);
    }
  }

  initNcclResources();

  init_state_ = InitializationState::INITIALIZED;
  TracingGuard tracingGuard(name_, comm_size_, "init", rank_, sequence_number_);

  TC_LOG(INFO, this) << "ProcessGroupNCCL initialized for rank: " << rank_;
}

void ProcessGroupNCCL::initNcclResources() {
  c10::cuda::CUDAGuard gpuGuard(device_);

  is_high_priority_stream_ = options_c10d_->is_high_priority_stream ||
      getCvarBool(::c10d::TORCH_NCCL_HIGH_PRIORITY, false);

  if (!internal_stream_) {
    internal_stream_.emplace(
        at::cuda::getStreamFromPool(is_high_priority_stream_, device_.index()));
  }

  if (!dependency_event_) {
    dependency_event_.emplace(cudaEventDisableTiming);
  }

  max_event_pool_size_ = kDefaultMaxEventPoolSize;

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commUserRank(nccl_comm_, &rank_),
      "NCCL User Rank failed");

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commCount(nccl_comm_, &comm_size_),
      "NCCL Count failed");

  if (!blocking_wait_ && !shutdown_) {
    timeout_thread_ = std::thread(&ProcessGroupNCCL::timeoutWatchdog, this);
  }

  attachMemoryHook();
  publishComm();
}

void ProcessGroupNCCL::initFromComm(
    ncclComm_t comm,
    at::Device device,
    std::shared_ptr<NcclApi> nccl_api) {
  device_ = device;
  nccl_api_ = std::move(nccl_api);
  nccl_comm_ = comm;
  initNcclResources();
  init_state_ = InitializationState::INITIALIZED;
  TracingGuard tracingGuard(name_, comm_size_, "init", rank_, sequence_number_);
  TC_LOG(INFO, this) << "ProcessGroupNCCL initialized from split for rank: "
                     << rank_;
}

void ProcessGroupNCCL::performNocolorSplit(at::Device device) {
  ensureInitialized(device);
  auto opts = Options::create(options_c10d_->is_high_priority_stream);
  opts->timeout = options_c10d_->timeout;
  opts->config = cloneNcclConfig(options_c10d_->config);
  split(store_, {}, opts);
}

c10::intrusive_ptr<::c10d::Backend> ProcessGroupNCCL::split(
    const c10::intrusive_ptr<::c10d::Store>& store,
    const std::vector<int>& ranks,
    const c10::intrusive_ptr<::c10d::Backend::Options>& opts) {
  // Validate the requested ranks (in range, no duplicates). Port of the
  // validation in TorchCommNCCL::split.
  std::unordered_set<int> seen;
  for (int r : ranks) {
    TORCH_CHECK(
        r >= 0 && r < getSize(),
        "ProcessGroupNCCL::split: invalid rank ",
        r,
        "; valid ranks are 0 to ",
        getSize() - 1);
    TORCH_CHECK(
        seen.insert(r).second,
        "ProcessGroupNCCL::split: rank ",
        r,
        " appears multiple times in ranks");
  }

  auto ncclOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
  TORCH_CHECK(
      ncclOpts != nullptr,
      "ProcessGroupNCCL::split: opts is not a nccl2 ProcessGroupNCCL::Options");

  // ncclCommSplit is collective over the parent communicator, so the parent
  // must be initialized before we split. All parent ranks call split(), so a
  // lazy bootstrap here stays in lockstep. Resolve a device the same way the
  // lazy collective path does.
  if (init_state_ != InitializationState::INITIALIZED) {
    at::Device dev = getBoundDeviceId().has_value()
        ? getBoundDeviceId().value()
        : at::Device(at::kCUDA, at::cuda::current_device());
    ensureInitialized(dev);
  }
  checkAndAbortIfTimedOutOrError();

  // Determine this rank's color and its rank within the child communicator.
  // color = lowest rank in the group (each rank joins exactly one group);
  // key = index within `ranks`, giving deterministic child rank ordering.
  // A rank not in `ranks` uses NCCL_SPLIT_NOCOLOR and produces no child comm.
  int color = NCCL_SPLIT_NOCOLOR;
  int newRank = -1;
  auto it = std::find(ranks.begin(), ranks.end(), getRank());
  if (it != ranks.end()) {
    color = *std::min_element(ranks.begin(), ranks.end());
    newRank = static_cast<int>(std::distance(ranks.begin(), it));
  }

  const std::string& name = ncclOpts->group_name;
  ncclConfig_t config =
      newRank == -1 ? ncclOpts->config : cloneNcclConfig(ncclOpts->config);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  config.commName = name.c_str();
#endif

  // Collective on the parent comm: every parent rank calls commSplit exactly
  // once, members with their color and non-members with NCCL_SPLIT_NOCOLOR.
  c10::cuda::CUDAGuard gpuGuard(device_);
  ncclComm_t new_comm = nullptr;
  const int key = newRank >= 0 ? newRank : getRank();
  auto splitStatus =
      nccl_api_->commSplit(nccl_comm_, color, key, &new_comm, &config);
  try {
    waitForNcclChildComm(
        *nccl_api_,
        nccl_comm_,
        &new_comm,
        splitStatus,
        newRank != -1,
        ncclOpts->timeout,
        "NCCL split failed");
  } catch (...) {
    comm_state_ = CommState::ERROR;
    nccl_comm_ = nullptr;
    throw;
  }

  if (newRank == -1) {
    // Non-member: no child communicator.
    return nullptr;
  }

  auto childOpts = Options::create(ncclOpts->is_high_priority_stream);
  childOpts->timeout = ncclOpts->timeout;
  childOpts->config = config;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  // commName above borrows `name`, which belongs to the Options handed to
  // split() -- a clone that dies with the caller's frame. Every consumer of a
  // stored config assigns commName from a live string before passing it to
  // NCCL (createNcclComm, reconfigure(), this function), so store no pointer
  // rather than one that outlives its string.
  childOpts->config.commName = nullptr;
#endif
  childOpts->group_name = ncclOpts->group_name;
  childOpts->group_desc = ncclOpts->group_desc;

  // The child's membership in world-rank terms, as stock's split() records it.
  // nccl2 never reads this itself; its one consumer is FlightRecorderHook,
  // which otherwise falls back to identity ranks. Map through this group's own
  // map, not the passed-in Options': an empty map means "spans the world in
  // rank order", and a merged group inherits its parent's shorter map, so
  // treat anything too short the same way instead of indexing out of bounds.
  const auto& parentRanks = options_c10d_->global_ranks_in_group;
  const bool parentSpansWorld =
      parentRanks.size() < static_cast<size_t>(getSize());
  std::vector<uint64_t> childRanks;
  childRanks.reserve(ranks.size());
  for (int r : ranks) {
    childRanks.push_back(
        parentSpansWorld ? static_cast<uint64_t>(r) : parentRanks[r]);
  }
  childOpts->global_ranks_in_group = std::move(childRanks);

  auto child = c10::make_intrusive<ProcessGroupNCCL>(
      store, newRank, static_cast<int>(ranks.size()), childOpts);
  child->initFromComm(new_comm, device_, nccl_api_);
  return c10::static_intrusive_pointer_cast<::c10d::Backend>(child);
}

void ProcessGroupNCCL::abort() {
  // User-initiated (backend.abort() / _abort_process_group()): tear the
  // communicator down and return. Never terminates the process, whatever
  // TORCH_NCCL_ASYNC_ERROR_HANDLING says -- that gate only covers failures the
  // watchdog detects on its own.
  TC_LOG(INFO, this) << "abort() requested on rank " << rank_
                     << "; aborting the NCCL communicator";
  if (options_c10d_->enable_reconfigure) {
    comm_state_ = CommState::ERROR;
    revokeNcclComm();
  } else {
    // Stop the watchdog before publishing the error: this failure is one the
    // caller asked for, so it must not trigger a process abort, and there is no
    // communicator left to watch either. Not done in reconfigurable mode, where
    // the comm is revoked rather than destroyed and the watchdog is needed
    // again after reconfigure(). The error is still published before the
    // teardown, so work in flight reports it instead of completing.
    stopWatchdog();
    comm_state_ = CommState::ERROR;
    abortNcclComm();
  }
}

void ProcessGroupNCCL::suspend() {
  checkInitialized();
  c10::cuda::CUDAGuard gpuGuard(device_);
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commSuspend(nccl_comm_, NCCL_SUSPEND_MEM),
      "NCCL Suspend failed (requires NCCL 2.29.7+)");
}

void ProcessGroupNCCL::resume() {
  checkInitialized();
  c10::cuda::CUDAGuard gpuGuard(device_);
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commResume(nccl_comm_),
      "NCCL Resume failed (requires NCCL 2.29.7+)");
}

std::unordered_map<std::string, uint64_t> ProcessGroupNCCL::getMemoryStats() {
  checkInitialized();
  c10::cuda::CUDAGuard gpuGuard(device_);
  // Stat indices follow ncclCommMemStat_t: suspend=0, suspended=1, persist=2,
  // total=3. Keys match ProcessGroupNCCL (the original backend).
  static constexpr std::array<std::pair<const char*, int>, 4> kStats = {
      {{"suspend", 0}, {"suspended", 1}, {"persist", 2}, {"total", 3}}};
  std::unordered_map<std::string, uint64_t> stats;
  for (const auto& [name, stat] : kStats) {
    uint64_t value = 0;
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commMemStats(nccl_comm_, stat, &value),
        "NCCL MemStats failed (requires NCCL 2.29.7+)");
    stats.emplace(name, value);
  }
  return stats;
}

::c10d::ErrorType ProcessGroupNCCL::getError() {
  switch (comm_state_.load()) {
    case CommState::TIMEOUT:
      return ::c10d::ErrorType::TIMEOUT;
    case CommState::ERROR:
      return ::c10d::ErrorType::COMM_ERROR;
    default:
      return ::c10d::ErrorType::SUCCESS;
  }
}

void ProcessGroupNCCL::finalize() {
  if (init_state_ == InitializationState::UNINITIALIZED) {
    throw std::runtime_error("ProcessGroupNCCL not initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("ProcessGroupNCCL already finalized");
  }
  init_state_ = InitializationState::FINALIZED;

  // Stop the watchdog first: draining the work queue below may surface a
  // timeout, which is a teardown result to report to the caller, not a reason
  // to terminate the process.
  stopWatchdog();

  // Wait for all pending work objects to complete and get final status
  auto work_status = workq_.finalize();

  if (work_status == WorkNCCL::WorkStatus::NOT_STARTED ||
      work_status == WorkNCCL::WorkStatus::INPROGRESS) {
    throw std::runtime_error(
        "WorkQ finalize returned in progress or not started state");
  }

  // Update comm_state_ based on the work status
  if (work_status == WorkNCCL::WorkStatus::TIMEDOUT) {
    comm_state_ = CommState::TIMEOUT;
    abortNcclComm();
    throw std::runtime_error("Work timed out during finalize");
  } else if (work_status == WorkNCCL::WorkStatus::ERROR) {
    comm_state_ = CommState::ERROR;
    if (!nccl_comm_) {
      throw std::runtime_error(
          "NCCL communicator was aborted after a previous error");
    }
    ncclResult_t asyncErr{};
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commGetAsyncError(nccl_comm_, &asyncErr),
        "failed to get async error");
    NCCLException ncclException(
        *nccl_api_, "NCCL Async Error", asyncErr, nccl_comm_);
    abortNcclComm();
    throw std::move(ncclException);
  }

  // Clean up event pool
  {
    std::lock_guard<std::mutex> lock(event_pool_mutex_);
    while (!event_pool_.empty()) {
      event_pool_.pop();
    }
  }

  barrier_buffer_.clear();

  dependency_event_.reset();
  internal_stream_.reset();

  // Destroy NCCL communicator
  // Note: If abortNcclComm() was called, nccl_comm_ is already nullptr and this
  // is skipped. We must not call commDestroy after commAbort per NCCL docs.
  if (nccl_comm_) {
    detachMemoryHook();
    retireComm();
    // Deregister comm from the CachingAllocator
    waitForNcclCompletion(
        *nccl_api_,
        nccl_comm_,
        nccl_api_->commDestroy(nccl_comm_),
        options_c10d_->timeout,
        "NCCL Destroy failed");
    nccl_comm_ = nullptr;
  }
}

void ProcessGroupNCCL::abortNcclComm() {
  detachMemoryHook();
  retireComm();
  if (nccl_comm_) {
    waitForNcclCompletion(
        *nccl_api_,
        nccl_comm_,
        nccl_api_->commAbort(nccl_comm_),
        options_c10d_->timeout,
        "NCCL Abort failed");
    nccl_comm_ = nullptr;
  }
}

void ProcessGroupNCCL::stopWatchdog() {
  shutdown_ = true;
  {
    std::lock_guard<std::mutex> lock(timeout_mutex_);
    timeout_cv_.notify_all();
  }
  if (timeout_thread_.joinable()) {
    // Joining from within the watchdog thread (the async-error path calls
    // abort()) would deadlock.
    if (std::this_thread::get_id() != timeout_thread_.get_id()) {
      timeout_thread_.join();
    } else {
      timeout_thread_.detach(); // NOLINT(facebook-hte-BadCall-detach)
    }
  }
}

void ProcessGroupNCCL::abortProcess(const std::string& reason) {
  // Never terminate the process in reconfigurable mode: callers fall back to
  // revoke + throw so the failure can be handled by reconfiguring.
  if (!SHOULD_TEAR_DOWN(async_error_handling_) ||
      options_c10d_->enable_reconfigure) {
    return;
  }
  TC_LOG(ERROR, this) << "Aborting process on rank " << rank_ << " due to "
                      << reason;
  runAbortHooks();
  ::abort();
}

void ProcessGroupNCCL::handleWatchdogFailure(const std::string& reason) {
  if (options_c10d_->enable_reconfigure) {
    revokeNcclComm();
    return;
  }

  if (SHOULD_CLEAN_UP(async_error_handling_)) {
    if (timeout_thread_.joinable() &&
        std::this_thread::get_id() == timeout_thread_.get_id()) {
      shutdown_ = true;
      timeout_cv_.notify_all();
    } else {
      stopWatchdog();
    }
    try {
      abortNcclComm();
    } catch (const std::exception& e) {
      TC_LOG(ERROR, this) << "Failed to clean up NCCL communicator after "
                          << reason << ": " << e.what();
      abortProcess(reason);
      return;
    }
  }
  abortProcess(reason);
}

void ProcessGroupNCCL::handleBlockingWaitFailure(
    WorkNCCL::WorkStatus status,
    int64_t reconfigure_uuid) {
  std::lock_guard reconfigureLock(reconfigure_mutex_);
  if (reconfigure_uuid_ != reconfigure_uuid) {
    return;
  }
  comm_state_ = status == WorkNCCL::WorkStatus::TIMEDOUT ? CommState::TIMEOUT
                                                         : CommState::ERROR;
  if (options_c10d_->enable_reconfigure) {
    revokeNcclComm();
  } else {
    abortNcclComm();
  }
}

void ProcessGroupNCCL::revokeNcclComm() {
  // Idempotent: the timeout watchdog and a synchronous collective may both
  // observe the same timeout and attempt a revoke. Run the teardown (abort
  // hooks, memory hook detach, commRevoke) at most once per communicator
  // generation. revoked_ is reset on reconfigure.
  if (revoked_.exchange(true)) {
    return;
  }
  TC_LOG(INFO, this) << "Calling abort hooks before commRevoke.";
  runAbortHooks();
  detachMemoryHook();
  retireComm();
  if (nccl_comm_) {
    // Best-effort: this may run on the timeout watchdog thread, so log instead
    // of throwing on failure (the communicator is already being torn down).
    NCCL_CHECK_IGNORE(
        nccl_api_, nccl_api_->commRevoke(nccl_comm_), "NCCL Revoke failed");
  }
}

int64_t ProcessGroupNCCL::getCommPtr() const {
  return reinterpret_cast<int64_t>(nccl_comm_);
}

void ProcessGroupNCCL::publishComm() {
  ::c10d::publishNCCLComm(
      getGroupUid(), reinterpret_cast<void*>(nccl_comm_), device_);
}

void ProcessGroupNCCL::retireComm() {
  ::c10d::retireNCCLComm(
      getGroupUid(), reinterpret_cast<void*>(nccl_comm_), device_);
}

// Point-to-Point Operations
c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::sendImpl(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "send", dst, sequence_number_, tensor, tensor);

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("send");

  // Wrap in ncclGroupStart/End so the kernel is enqueued on the stream before
  // we record the end event. Without the group wrapper, a non-blocking NCCL
  // comm may defer the kernel launch past the end-event record, so the event
  // can fire before the transfer completes and work.wait() returns with
  // stale/partial data. Matches batch_op_issue and stock
  // ProcessGroupNCCL::pointToPoint. (TorchComms fix D109625550.)
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");
  try {
    NCCL_CHECK_NONBLOCKING(
        nccl_api_,
        nccl_comm_,
        nccl_api_->send(
            tensor.data_ptr(),
            tensor.numel(),
            getNcclDataType(tensor),
            dst,
            nccl_comm_,
            stream),
        "NCCL Send failed");
  } catch (...) {
    // Close the group even on failure so the comm is not left mid-group for
    // subsequent operations on this thread. groupEnd's own error is only logged
    // since we are already propagating the original error.
    NCCL_CHECK_IGNORE(nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
    throw;
  }
  waitForNcclOperation(nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::recvImpl(
    at::Tensor& tensor,
    int src,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "recv", src, sequence_number_, tensor, tensor);

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("recv");

  // Wrap in ncclGroupStart/End -- see sendImpl comment for rationale.
  // (TorchComms fix D109625550.)
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");
  try {
    NCCL_CHECK_NONBLOCKING(
        nccl_api_,
        nccl_comm_,
        nccl_api_->recv(
            tensor.data_ptr(),
            tensor.numel(),
            getNcclDataType(tensor),
            src,
            nccl_comm_,
            stream),
        "NCCL Recv failed");
  } catch (...) {
    // Close the group even on failure so the comm is not left mid-group for
    // subsequent operations on this thread. groupEnd's own error is only logged
    // since we are already propagating the original error.
    NCCL_CHECK_IGNORE(nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
    throw;
  }
  waitForNcclOperation(nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Batch P2P Operations
c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  if (ops.empty()) {
    throw std::runtime_error("Cannot issue empty batch operation");
  }

  // Collect input and output tensors for work tracking
  std::vector<at::Tensor> input_tensors;
  std::vector<at::Tensor> output_tensors;

  for (const auto& op : ops) {
    checkTensorDevice(op.tensor);
    if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
      at::Tensor tensor = op.tensor;
      ensureTensorContiguous(tensor);
      input_tensors.push_back(tensor);
    } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
      at::Tensor tensor = op.tensor;
      ensureTensorContiguous(tensor);
      output_tensors.push_back(tensor);
    } else {
      throw std::runtime_error("Unknown op type");
    }
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "batch_op_issue",
      rank_,
      sequence_number_,
      input_tensors,
      output_tensors);

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = createWork(stream, timeout, input_tensors);

  // Record start event before NCCL operations
  work->recordStart("batch_op_issue");

  // Start NCCL group for batched operations
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  try {
    // Issue each operation individually
    for (const auto& op : ops) {
      if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
        NCCL_CHECK_NONBLOCKING(
            nccl_api_,
            nccl_comm_,
            nccl_api_->send(
                op.tensor.data_ptr(),
                op.tensor.numel(),
                getNcclDataType(op.tensor),
                op.peer,
                nccl_comm_,
                stream),
            "NCCL Send failed in batch operation");
      } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
        NCCL_CHECK_NONBLOCKING(
            nccl_api_,
            nccl_comm_,
            nccl_api_->recv(
                op.tensor.data_ptr(),
                op.tensor.numel(),
                getNcclDataType(op.tensor),
                op.peer,
                nccl_comm_,
                stream),
            "NCCL Recv failed in batch operation");
      }
    }
  } catch (...) {
    NCCL_CHECK_IGNORE(nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
    throw;
  }

  // End NCCL group
  waitForNcclOperation(nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Collective Operations
c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::broadcastImpl(
    at::Tensor& tensor,
    int root,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "broadcast", root, sequence_number_, tensor, tensor);

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);

  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("broadcast");

  waitForNcclOperation(
      nccl_api_->bcast(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          root,
          nccl_comm_,
          stream),
      timeout,
      "NCCL Broadcast failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::all_reduce(
    at::Tensor& tensor,
    const ::c10d::ReduceOp& op,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "all_reduce", rank_, sequence_number_, tensor, tensor);

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("all_reduce");

  const auto dataType = getNcclDataType(tensor);
  waitForNcclOperation(
      nccl_api_->allReduce(
          tensor.data_ptr(),
          tensor.data_ptr(), // In-place operation
          tensor.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, tensor),
          nccl_comm_,
          stream),
      timeout,
      "NCCL AllReduce failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::reduceImpl(
    const at::Tensor& tensor,
    int root,
    const ::c10d::ReduceOp& op,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkTensorDevice(tensor);

  TracingGuard tracingGuard(
      name_, comm_size_, "reduce", root, sequence_number_, tensor, tensor);

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("reduce");

  const auto dataType = getNcclDataType(tensor);
  waitForNcclOperation(
      nccl_api_->reduce(
          tensor.data_ptr(),
          rank_ == root ? tensor.data_ptr() : nullptr,
          tensor.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, tensor),
          root,
          nccl_comm_,
          stream),
      timeout,
      "NCCL Reduce failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  if (tensor_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "tensor_list size must equal comm_size for all_gather");
  }

  // Ensure input tensor is contiguous
  ensureTensorContiguous(tensor);

  // Mixed-device outputs are staged on the communicator's device.
  std::vector<at::Tensor> local_outputs;
  local_outputs.reserve(tensor_list.size());
  bool needs_staging = false;
  for (const auto& t : tensor_list) {
    ensureTensorContiguous(t);
    TORCH_CHECK(
        t.device().is_cuda(),
        "Expected all_gather output tensor on CUDA but found tensor on ",
        t.device());
  }
  TORCH_CHECK(
      tensor_list[rank_].numel() == tensor.numel(),
      "Input tensor must have the same size as the output tensor for its rank");

  checkTensorDevice(tensor);
  checkSameDtype(tensor, tensor_list);
  for (const auto& output : tensor_list) {
    if (output.device() == device_) {
      local_outputs.push_back(output);
    } else {
      local_outputs.push_back(at::empty(
          output.sizes(),
          tensor.options().memory_format(at::MemoryFormat::Contiguous)));
      needs_staging = true;
    }
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "all_gather",
      rank_,
      sequence_number_,
      tensor_list,
      {tensor});

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto operation_stream =
      at::cuda::getStreamFromExternal(stream, device_.index());
  for (size_t i = 0; i < local_outputs.size(); ++i) {
    if (local_outputs[i].device() != tensor_list[i].device()) {
      c10::cuda::CUDACachingAllocator::recordStream(
          local_outputs[i].storage().data_ptr(), operation_stream);
    }
  }
  auto work = async_op ? createWork(stream, timeout, tensor)
                       : createWork(stream, timeout);

  work->recordStart("all_gather");

  // Use multiple broadcast operations for all_gather
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  try {
    for (int i = 0; i < comm_size_; ++i) {
      const auto& input = i == rank_ ? tensor : local_outputs[i];
      NCCL_CHECK_NONBLOCKING(
          nccl_api_,
          nccl_comm_,
          nccl_api_->broadcast(
              input.data_ptr(),
              local_outputs[i].data_ptr(),
              local_outputs[i].numel(),
              getNcclDataType(local_outputs[i]),
              i,
              nccl_comm_,
              stream),
          "NCCL Broadcast failed in all_gather");
    }
  } catch (...) {
    NCCL_CHECK_IGNORE(nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
    throw;
  }

  waitForNcclOperation(nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");

  if (needs_staging) {
    at::cuda::CUDAStreamGuard stream_guard(operation_stream);
    for (size_t i = 0; i < tensor_list.size(); ++i) {
      if (local_outputs[i].device() != tensor_list[i].device()) {
        tensor_list[i].copy_(local_outputs[i], true);
      }
    }
  }

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::allGatherSingleImpl(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkTensorDevice(output);
  checkTensorDevice(input);
  checkSameDtype(input, output);

  if (output.numel() != input.numel() * comm_size_) {
    throw std::runtime_error(
        "Output tensor size must be input_size * comm_size for allGatherSingleImpl");
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "allGatherSingleImpl",
      rank_,
      sequence_number_,
      input,
      output);

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input)
                       : createWork(stream, timeout);

  work->recordStart("allGatherSingleImpl");

  waitForNcclOperation(
      nccl_api_->allGather(
          input.data_ptr(),
          output.data_ptr(),
          input.numel(),
          getNcclDataType(input),
          nccl_comm_,
          stream),
      timeout,
      "NCCL AllGather failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ::c10d::ReduceOp& op,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);

  if (input_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "input_list size must equal comm_size for reduce_scatter");
  }

  // Check that all input tensors are contiguous.
  for (const auto& t : input_list) {
    ensureTensorContiguous(t);
  }
  TORCH_CHECK(
      input_list[rank_].numel() == output.numel(),
      "Output tensor must have the same size as the input tensor for its rank");

  checkTensorsDevice(input_list);
  checkTensorDevice(output);
  checkSameDtype(output, input_list);

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "reduce_scatter",
      rank_,
      sequence_number_,
      input_list,
      {output});

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input_list)
                       : createWork(stream, timeout);

  work->recordStart("reduce_scatter");

  // Use multiple reduce operations for reduce_scatter
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  try {
    for (int i = 0; i < comm_size_; ++i) {
      const auto dataType = getNcclDataType(input_list[i]);
      if (i == rank_) {
        // This rank receives the reduced result
        NCCL_CHECK_NONBLOCKING(
            nccl_api_,
            nccl_comm_,
            nccl_api_->reduce(
                input_list[i].data_ptr(),
                output.data_ptr(),
                output.numel(),
                dataType,
                getNcclReduceOp(op, nccl_comm_, input_list[i]),
                i,
                nccl_comm_,
                stream),
            "NCCL Reduce failed in reduce_scatter");
      } else {
        // Other ranks contribute to the reduction
        NCCL_CHECK_NONBLOCKING(
            nccl_api_,
            nccl_comm_,
            nccl_api_->reduce(
                input_list[i].data_ptr(),
                nullptr, // Non-root ranks don't receive
                input_list[i].numel(),
                dataType,
                getNcclReduceOp(op, nccl_comm_, input_list[i]),
                i,
                nccl_comm_,
                stream),
            "NCCL Reduce failed in reduce_scatter");
      }
    }
  } catch (...) {
    NCCL_CHECK_IGNORE(nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
    throw;
  }

  waitForNcclOperation(nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::reduceScatterSingleImpl(
    at::Tensor& output,
    const at::Tensor& input,
    const ::c10d::ReduceOp& op,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkTensorDevice(output);
  checkTensorDevice(input);
  checkSameDtype(input, output);

  if (input.numel() != output.numel() * comm_size_) {
    throw std::runtime_error(
        "Input tensor size must be output_size * comm_size for reduceScatterSingleImpl");
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "reduceScatterSingleImpl",
      rank_,
      sequence_number_,
      input,
      output);

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("reduceScatterSingleImpl");

  const auto dataType = getNcclDataType(input);
  waitForNcclOperation(
      nccl_api_->reduceScatter(
          input.data_ptr(),
          output.data_ptr(),
          output.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, input),
          nccl_comm_,
          stream),
      timeout,
      "NCCL ReduceScatter failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::allToAllSingleImpl(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkTensorDevice(output);
  checkTensorDevice(input);
  checkSameDtype(input, output);

  if (input.numel() != output.numel()) {
    throw std::runtime_error(
        "Input and output tensors must have same size for allToAllSingleImpl");
  }

  if (input.numel() % comm_size_ != 0) {
    throw std::runtime_error(
        "Tensor size must be divisible by comm_size for allToAllSingleImpl");
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "allToAllSingleImpl",
      rank_,
      sequence_number_,
      input,
      output);

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("allToAllSingleImpl");

  size_t chunk_size = input.numel() / comm_size_;
  const auto data_type = getNcclDataType(input);

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  waitForNcclOperation(
      nccl_api_->allToAll(
          input.data_ptr(),
          output.data_ptr(),
          chunk_size,
          data_type,
          nccl_comm_,
          stream),
      timeout,
      "NCCL AllToAll failed");
#else
  size_t offset = chunk_size * wordSize(data_type);
  char* sptr = static_cast<char*>(input.data_ptr());
  char* rptr = static_cast<char*>(output.data_ptr());
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  try {
    for (int i = 0; i < comm_size_; ++i) {
      // Send to rank i
      NCCL_CHECK_NONBLOCKING(
          nccl_api_,
          nccl_comm_,
          nccl_api_->send(
              sptr + i * offset, chunk_size, data_type, i, nccl_comm_, stream),
          "NCCL Send failed in allToAllSingleImpl");

      // Receive from rank i
      NCCL_CHECK_NONBLOCKING(
          nccl_api_,
          nccl_comm_,
          nccl_api_->recv(
              rptr + i * offset, chunk_size, data_type, i, nccl_comm_, stream),
          "NCCL Recv failed in allToAllSingleImpl");
    }
  } catch (...) {
    NCCL_CHECK_IGNORE(nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
    throw;
  }

  waitForNcclOperation(nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");
#endif

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkTensorDevice(output);
  checkTensorDevice(input);
  checkSameDtype(input, output);

  // Validate split sizes vectors
  if (input_split_sizes.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "input_split_sizes length must equal comm_size for all_to_all_v_single");
  }

  if (output_split_sizes.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "output_split_sizes length must equal comm_size for all_to_all_v_single");
  }

  // Validate that split sizes sum does not exceed tensor dimensions
  uint64_t input_total = 0;
  uint64_t output_total = 0;
  for (int i = 0; i < comm_size_; ++i) {
    input_total += input_split_sizes[i];
    output_total += output_split_sizes[i];
  }

  if (input_total > static_cast<uint64_t>(input.size(0))) {
    throw std::runtime_error(
        "Sum of input_split_sizes exceeds input tensor size for all_to_all_v_single");
  }

  if (output_total > static_cast<uint64_t>(output.size(0))) {
    throw std::runtime_error(
        "Sum of output_split_sizes exceeds output tensor size for all_to_all_v_single");
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "all_to_all_v_single",
      rank_,
      sequence_number_,
      input,
      output);

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input)
                       : createWork(stream, timeout);

  // Record start event before NCCL operation
  work->recordStart("all_to_all_v_single");

  // Convert split sizes to arrays and calculate displacements
  std::vector<size_t> sendcounts(comm_size_);
  std::vector<size_t> recvcounts(comm_size_);
  std::vector<size_t> senddispls(comm_size_);
  std::vector<size_t> recvdispls(comm_size_);

  // Calculate the number of elements per slice along the first dimension
  // For a tensor with shape [N, D1, D2, ..., Dk], each slice of size S along
  // dim 0 contains S * D1 * D2 * ... * Dk elements
  // Use input tensor for send counts and output tensor for recv counts
  size_t send_elements_per_slice =
      input.numel() ? input.numel() / input.size(0) : 0;
  size_t recv_elements_per_slice =
      output.numel() ? output.numel() / output.size(0) : 0;
  const auto data_type = getNcclDataType(input);
  const size_t type_size = wordSize(data_type);

  size_t sendoffset = 0;
  size_t recvoffset = 0;
  for (int i = 0; i < comm_size_; ++i) {
    sendcounts[i] = input_split_sizes[i] * send_elements_per_slice;
    recvcounts[i] = output_split_sizes[i] * recv_elements_per_slice;
    senddispls[i] = sendoffset;
    recvdispls[i] = recvoffset;
    sendoffset += sendcounts[i];
    recvoffset += recvcounts[i];
  }

  char* sptr = static_cast<char*>(input.data_ptr());
  char* rptr = static_cast<char*>(output.data_ptr());

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  try {
    for (int i = 0; i < comm_size_; ++i) {
      NCCL_CHECK_NONBLOCKING(
          nccl_api_,
          nccl_comm_,
          nccl_api_->send(
              sptr + senddispls[i] * type_size,
              sendcounts[i],
              data_type,
              i,
              nccl_comm_,
              stream),
          "NCCL Send failed in all_to_all_v_single");
      NCCL_CHECK_NONBLOCKING(
          nccl_api_,
          nccl_comm_,
          nccl_api_->recv(
              rptr + recvdispls[i] * type_size,
              recvcounts[i],
              data_type,
              i,
              nccl_comm_,
              stream),
          "NCCL Recv failed in all_to_all_v_single");
    }
  } catch (...) {
    NCCL_CHECK_IGNORE(nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
    throw;
  }

  waitForNcclOperation(nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  checkTensorsDevice(output_tensor_list);
  checkTensorsDevice(input_tensor_list);
  if (output_tensor_list.size() != static_cast<size_t>(comm_size_) ||
      input_tensor_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "Tensor list sizes must equal comm_size for all_to_all");
  }

  // Validate all tensors
  for (int i = 0; i < comm_size_; ++i) {
    ensureTensorContiguous(input_tensor_list[i]);
    ensureTensorContiguous(output_tensor_list[i]);
    checkSameDtype(input_tensor_list[0], input_tensor_list[i]);
    checkSameDtype(input_tensor_list[0], output_tensor_list[i]);
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "all_to_all",
      rank_,
      sequence_number_,
      input_tensor_list,
      output_tensor_list);

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op ? createWork(stream, timeout, input_tensor_list)
                       : createWork(stream, timeout);

  // Record start event before NCCL operations
  work->recordStart("all_to_all");

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  try {
    for (int i = 0; i < comm_size_; ++i) {
      // Send to rank i
      NCCL_CHECK_NONBLOCKING(
          nccl_api_,
          nccl_comm_,
          nccl_api_->send(
              input_tensor_list[i].data_ptr(),
              input_tensor_list[i].numel(),
              getNcclDataType(input_tensor_list[i]),
              i,
              nccl_comm_,
              stream),
          "NCCL Send failed in all_to_all");

      // Receive from rank i
      NCCL_CHECK_NONBLOCKING(
          nccl_api_,
          nccl_comm_,
          nccl_api_->recv(
              output_tensor_list[i].data_ptr(),
              output_tensor_list[i].numel(),
              getNcclDataType(output_tensor_list[i]),
              i,
              nccl_comm_,
              stream),
          "NCCL Recv failed in all_to_all");
    }
  } catch (...) {
    NCCL_CHECK_IGNORE(nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
    throw;
  }

  waitForNcclOperation(nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::barrierImpl(
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  // Allocated here rather than in initNcclResources() so that merely creating a
  // process group never touches the CUDA caching allocator. Eager init runs
  // before anything has initialized CUDA (init_process_group(device_id=...) on
  // a process that has made no torch.cuda call), and the allocator is brought
  // up lazily, so allocating there tripped "Allocator not initialized for
  // device N". lazyInitDevice() is what ATen's own allocating paths call, and
  // is a no-op once CUDA is up.
  if (!barrier_buffer_) {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    c10::cuda::CUDAGuard gpuGuard(device_);
    barrier_buffer_ =
        c10::cuda::CUDACachingAllocator::get()->allocate(sizeof(float));
  }

  TracingGuard tracingGuard(
      name_, comm_size_, "barrier", rank_, sequence_number_);
  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = createWork(stream, timeout);

  // A synchronous barrier host-blocks the CPU thread in synchronizeInternal(),
  // matching stock ProcessGroupNCCL; async barriers stay stream-ordered.
  work->hostBlocking_ = !async_op;

  // Record start event before NCCL operation
  work->recordStart("barrier");

  // Use pre-allocated CUDA buffer for barrier
  waitForNcclOperation(
      nccl_api_->allReduce(
          barrier_buffer_.get(),
          barrier_buffer_.get(),
          1,
          ncclFloat32,
          ncclSum,
          nccl_comm_,
          stream),
      timeout,
      "NCCL Barrier failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::scatterImpl(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output_tensor);
  checkTensorDevice(output_tensor);
  checkTensorsDevice(input_tensor_list);

  // Only the root rank needs valid input tensors
  if (rank_ == root) {
    if (input_tensor_list.size() != static_cast<size_t>(comm_size_)) {
      throw std::runtime_error(
          "input_tensor_list size must equal comm_size for scatter");
    }

    for (const auto& t : input_tensor_list) {
      ensureTensorContiguous(t);
      checkSameDtype(output_tensor, t);
      if (t.numel() != output_tensor.numel()) {
        throw std::runtime_error(
            "All input tensors must have same size as output tensor");
      }
    }
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "scatter",
      root,
      sequence_number_,
      input_tensor_list,
      {output_tensor});

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> input_tensors;
  if (async_op && rank_ == root) {
    input_tensors = input_tensor_list;
  }
  auto work = createWork(stream, timeout, input_tensors);

  // Record start event before NCCL operations
  work->recordStart("scatter");

  // Implement scatter using point-to-point operations
  if (rank_ == root) {
    // Root sends to all ranks (except itself)
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->groupStart(),
        "NCCL GroupStart failed");
    try {
      for (int i = 0; i < comm_size_; ++i) {
        if (i != root) {
          NCCL_CHECK_NONBLOCKING(
              nccl_api_,
              nccl_comm_,
              nccl_api_->send(
                  input_tensor_list[i].data_ptr(),
                  input_tensor_list[i].numel(),
                  getNcclDataType(input_tensor_list[i]),
                  i,
                  nccl_comm_,
                  stream),
              "NCCL Send failed in scatter");
        }
      }
    } catch (...) {
      NCCL_CHECK_IGNORE(
          nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
      throw;
    }
    waitForNcclOperation(
        nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");

    at::cuda::CUDAStreamGuard stream_guard(
        at::cuda::getStreamFromExternal(stream, device_.index()));
    output_tensor.copy_(input_tensor_list[root], true);
  } else {
    // Non-root ranks receive from root
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->groupStart(),
        "NCCL GroupStart failed");
    try {
      NCCL_CHECK_NONBLOCKING(
          nccl_api_,
          nccl_comm_,
          nccl_api_->recv(
              output_tensor.data_ptr(),
              output_tensor.numel(),
              getNcclDataType(output_tensor),
              root,
              nccl_comm_,
              stream),
          "NCCL Recv failed in scatter");
    } catch (...) {
      NCCL_CHECK_IGNORE(
          nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
      throw;
    }
    waitForNcclOperation(
        nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");
  }

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<WorkNCCL> ProcessGroupNCCL::gatherImpl(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    std::chrono::milliseconds timeout) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(input_tensor);
  checkTensorDevice(input_tensor);
  checkTensorsDevice(output_tensor_list);

  // Only the root rank needs valid output tensors
  if (rank_ == root) {
    if (output_tensor_list.size() != static_cast<size_t>(comm_size_)) {
      throw std::runtime_error(
          "output_tensor_list size must equal comm_size for gather");
    }

    for (const auto& t : output_tensor_list) {
      ensureTensorContiguous(t);
      checkSameDtype(input_tensor, t);
      if (t.numel() != input_tensor.numel()) {
        throw std::runtime_error(
            "All output tensors must have same size as input tensor");
      }
    }
  }

  TracingGuard tracingGuard(
      name_,
      comm_size_,
      "gather",
      root,
      sequence_number_,
      {input_tensor},
      output_tensor_list);

  c10::cuda::CUDAGuard device_guard(device_);
  cudaStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> output_tensors;
  if (rank_ == root) {
    output_tensors = output_tensor_list;
  }
  auto work = async_op ? createWork(stream, timeout, input_tensor)
                       : createWork(stream, timeout);

  // Record start event before NCCL operations
  work->recordStart("gather");

  if (rank_ == root) {
    // Root receives from all ranks (except itself)
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->groupStart(),
        "NCCL GroupStart failed");
    try {
      for (int i = 0; i < comm_size_; ++i) {
        if (i != root) {
          NCCL_CHECK_NONBLOCKING(
              nccl_api_,
              nccl_comm_,
              nccl_api_->recv(
                  output_tensor_list[i].data_ptr(),
                  output_tensor_list[i].numel(),
                  getNcclDataType(output_tensor_list[i]),
                  i,
                  nccl_comm_,
                  stream),
              "NCCL Recv failed in gather");
        }
      }
    } catch (...) {
      NCCL_CHECK_IGNORE(
          nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
      throw;
    }
    waitForNcclOperation(
        nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");

    at::cuda::CUDAStreamGuard stream_guard(
        at::cuda::getStreamFromExternal(stream, device_.index()));
    output_tensor_list[root].copy_(input_tensor, true);
  } else {
    // Non-root ranks send to root
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->groupStart(),
        "NCCL GroupStart failed");
    try {
      NCCL_CHECK_NONBLOCKING(
          nccl_api_,
          nccl_comm_,
          nccl_api_->send(
              input_tensor.data_ptr(),
              input_tensor.numel(),
              getNcclDataType(input_tensor),
              root,
              nccl_comm_,
              stream),
          "NCCL Send failed in gather");
    } catch (...) {
      NCCL_CHECK_IGNORE(
          nccl_api_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");
      throw;
    }
    waitForNcclOperation(
        nccl_api_->groupEnd(), timeout, "NCCL GroupEnd failed");
  }

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

NCCLException::NCCLException(
    NcclApi& nccl_api,
    const std::string& message,
    ncclResult_t result,
    ncclComm_t comm)
    : message_(fmt::format(
          "{}: {} \nNCCL Last Error: {}",
          message,
          nccl_api.getErrorString(result),
          nccl_api.getLastError(comm))),
      result_(result) {}

const char* NCCLException::what() const noexcept {
  return message_.c_str();
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
