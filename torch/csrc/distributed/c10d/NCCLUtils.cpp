#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>

#include <c10/util/env.h>

#ifdef USE_C10D_NCCL
#include <mutex>
#include <vector>

namespace c10d {

NCCLComm::NCCLComm(ncclComm_t ncclComm) : ncclComm_(ncclComm) {}

NCCLComm::~NCCLComm() noexcept {
  // (kwen2501) Making CUDA/NCCL calls in this destructor can hit CUDA driver
  // shutdown error if CUDA context has exited first. Thus, we are not
  // destroying or aborting NCCL communicators here. We just detect and warn
  // about the risk of memory leak. Normally, a user would have called
  // `destroy_process_group` or `abort_process_group`, and such risk would be
  // avoided.
  LockType lock(mutex_);
  if (ncclComm_ && initialized_ && !aborted_) {
    TORCH_WARN_ONCE(
        "WARNING: NCCL communicator hasn't been destroyed. This may cause "
        "memory leaks. To avoid the risk, you can call `destroy_process_group` "
        "during normal exit or `_abort_process_group` when handling failures.")
  }
}

// NOLINTNEXTLINE(*-noexcept-move-*)
NCCLComm::NCCLComm(NCCLComm&& other) {
  // Using other's lock, as it reads other's states
  // Can not use this.mutex_, as this object is being constructed.
  LockType lock(other.mutex_);
  std::swap(ncclComm_, other.ncclComm_);
  std::swap(aborted_, other.aborted_);
  std::swap(ncclAsyncErr_, other.ncclAsyncErr_);
  std::swap(initialized_, other.initialized_);
  std::swap(nonBlocking_, other.nonBlocking_);
  std::swap(deviceIndex_, other.deviceIndex_);
}

ncclUniqueId NCCLComm::getNcclId() {
  return ncclId_;
}

std::shared_ptr<NCCLComm> NCCLComm::create(
    int numRanks,
    int rank,
    ncclUniqueId commId,
    at::DeviceIndex deviceIndex) {
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex);
  auto comm = std::make_shared<NCCLComm>();
  C10D_NCCL_CHECK(
      ncclCommInitRank(&(comm->ncclComm_), numRanks, commId, rank),
      std::nullopt);
  comm->ncclId_ = commId;
  comm->rank_ = rank;
  comm->deviceIndex_ = deviceIndex;
  comm->initialized_ = true;
  // Old style comm is always blocking.
  comm->nonBlocking_ = false;
  return comm;
}

#ifdef NCCL_HAS_COMM_NONBLOCKING
std::shared_ptr<NCCLComm> NCCLComm::create(
    int numRanks,
    int rank,
    ncclUniqueId commId,
    at::DeviceIndex deviceIndex,
    ncclConfig_t& config) {
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex);
  auto comm = std::make_shared<NCCLComm>();
  comm->nonBlocking_ = config.blocking == 0;
  LOG(INFO) << "Rank " << rank << ": creating NCCL communicator with mode: "
            << (comm->nonBlocking_ ? "nonblocking" : "blocking");
  C10D_NCCL_CHECK_NONBLOCKING(
      ncclCommInitRankConfig(
          &(comm->ncclComm_), numRanks, commId, rank, &config),
      std::nullopt);
  comm->ncclId_ = commId;
  comm->rank_ = rank;
  comm->deviceIndex_ = deviceIndex;
  // Under blocking mode, comm is initialized immediately after NCCL init
  // returns; Under nonblocking mode, we check whether comm is initialized the
  // *next* time ncclComm_ is accessed.
  comm->initialized_ = !comm->nonBlocking_;
  return comm;
}
#endif

ncclComm_t NCCLComm::getNcclComm() {
  LockType lock(mutex_);
  if (aborted_) {
    auto commFailureMsg = commFailureReason_ != std::nullopt
        ? c10::str(" Original reason for failure was: ", *commFailureReason_)
        : "";
    TORCH_CHECK_WITH(
        DistBackendError,
        false,
        c10::str(
            "NCCL communicator was aborted on rank ",
            rank_,
            ". ",
            commFailureMsg));
  }
  // In non-blocking mode, ensure comm is ready.
  if (nonBlocking_) {
    // Wait with long interval if communicator is being initialized.
    bool longInterval = !initialized_;
    waitReady(longInterval);
    // ncclComm_ should be initialized by now
  }
  if (!initialized_) {
    // TODO: see if we can consolidate other `initialized_` flipping here.
    // Maintaining it elsewhere is some work.
    initialized_ = true;
    LOG(INFO) << "Rank " << rank_ << ": NCCL communicator " << repr()
              << " is initialized.";
  }
  return ncclComm_;
}

// Wait for the communicator to be ready. This is a blocking function.
// Arguments:
//   longInterval: if true, wait with sleep of an interval; otherwise, wait
//   with `sched_yield` which is faster (but acquires CPU more frequently).
void NCCLComm::waitReady(bool longInterval) {
  LockType lock(mutex_);
  if (aborted_)
    return;
  // If timeout is reached, throw an exception.
  if (longInterval) {
    C10D_NCCL_CHECK_TIMEOUT_SLEEP(ncclInProgress, ncclComm_, std::nullopt);
  } else {
    C10D_NCCL_CHECK_TIMEOUT(ncclInProgress, ncclComm_, std::nullopt);
  }
}

std::optional<std::string> NCCLComm::getNcclCommFailureReason() const {
  LockType lock(mutex_);
  return commFailureReason_;
}

// TODO: why do we have `!defined(FBCODE_CAFFE2)` here?
#if defined(NCCL_HAS_COMM_SPLIT) && !defined(FBCODE_CAFFE2)
// last argument to split() API is not used to support
// multiple implementations
std::shared_ptr<NCCLComm> NCCLComm::split(
    NCCLComm* source,
    int color_id,
    int rank,
    ncclConfig_t& config,
    std::vector<uint64_t>& ranks_ull) {
  TORCH_CHECK(
      color_id >= NCCL_SPLIT_NOCOLOR,
      "Color must be a non-negative value or NCCL_SPLIT_NOCOLOR (-1)"
      ", but got ",
      color_id);
  LOG(INFO) << "Rank " << source->rank_ << ": split from parent comm "
            << source->repr() << " with color_id " << color_id << " and rank "
            << rank;
  at::cuda::OptionalCUDAGuard gpuGuard(source->deviceIndex_);
  auto comm = std::make_shared<NCCLComm>();
  // This call will block until the source communicator is initialized
  auto sourceComm = source->getNcclComm();
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(
      ncclCommSplit(sourceComm, color_id, rank, &(comm->ncclComm_), &config),
      std::nullopt);
#else
  // After calling ncclCommSplit in non-blocking mode, we should wait for the
  // source communicator to be out of ncclInProgress state.
  // Reason 1:
  //   it's unsafe to call new operations on the parent comm while it's in
  //   ncclInProgress state.
  // Reason 2:
  //   as of NCCL 2.23, the ptr value of child comm will not be filled until the
  //   state of parent comm is ncclSuccess. This may change in the future. See:
  //   https://github.com/NVIDIA/nccl/issues/1472
  C10D_NCCL_CHECK_TIMEOUT_SLEEP(
      ncclCommSplit(sourceComm, color_id, rank, &(comm->ncclComm_), &config),
      sourceComm, // wait on parent comm
      std::nullopt);
  if (color_id >= 0) {
    // Waiting for parent comm above still does not seem to guarantee the child
    // comm ptr is valid. Therefore we add a manual wait here for safety.
    // TODO: remove this wait after NCCL fix the semantics.
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = nccl_nonblocking_timeout();
    while (!comm->ncclComm_) {
      C10D_CHECK_TIMEOUT(startTime, timeout);
      C10D_SCHED_SLEEP();
    }
  }
  // comm->ncclComm_ should have valid ptr by now, but not necessarily
  // initialized. Rely on getNcclComm() to wait for its initialization.
#endif
  ++source->ncclCommSplitCounter_;
  comm->rank_ = rank;
  // Child comm should be on the same device as parent comm
  comm->deviceIndex_ = source->deviceIndex_;
  comm->nonBlocking_ = config.blocking == 0;
  LOG(INFO) << "Rank " << source->rank_ << ": created child comm "
            << comm->repr() << " with color_id " << color_id;
  return comm;
}
#endif

void NCCLComm::finalize() {
  LockType lock(mutex_);
  if (aborted_) {
    LOG(INFO) << "Rank " << rank_
              << ": NCCL communicator already Invalidated. Skip finalize.";
    return;
  }
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
  auto comm = getNcclComm();
  C10D_NCCL_CHECK_NONBLOCKING(ncclCommFinalize(comm), std::nullopt);
}

void NCCLComm::destroy() {
  LockType lock(mutex_);
  if (aborted_) {
    LOG(INFO) << "Rank " << rank_
              << ": NCCL communicator already Invalidated. Skip destroy.";
    return;
  }
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
  auto comm = getNcclComm();
  C10D_NCCL_CHECK(ncclCommDestroy(comm), std::nullopt);
  // Poison future getNcclComm
  aborted_ = true;
}

void NCCLComm::abort(std::optional<std::string> commFailureReason) {
  LockType lock(mutex_);
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
#ifdef ENABLE_NCCL_ERROR_CHECKING
  if (aborted_ && !initialized_) {
    // Should not abort twice.
    return;
  }

#ifdef NCCL_HAS_COMM_REGISTER
  // Deregister all registered segments before aborting.
  for (auto& it : registeredSegmentHandles_) {
    void* handle = it.second;
    C10D_NCCL_CHECK(
        ::ncclCommDeregister(ncclComm_, handle),
        c10::str(
            "Failed to deregister segment handle ",
            handle,
            " on ncclComm_ ",
            ncclComm_));
  }
  registeredSegmentHandles_.clear();
#endif

  // Set true failure reason if provided by ProcessGroupNCCL (e.g. work
  // timeout)
  commFailureReason_ = commFailureReason;
  LOG(INFO) << "Aborting ncclComm_ " << ncclComm_ << " with reason: "
            << (commFailureReason ? *commFailureReason
                                  : "No abort reason provided.");
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(::ncclCommAbort(ncclComm_), commFailureReason_);
#else
  C10D_NCCL_CHECK_TIMEOUT(
      ::ncclCommAbort(ncclComm_), ncclComm_, commFailureReason_);
#endif
  aborted_ = true;
  ncclComm_ = nullptr;

  // Set an appropriate error so that we avoid using the communicator.
  if (ncclAsyncErr_ == ncclSuccess) {
    ncclAsyncErr_ = ncclSystemError;
  }
#else
  // This is a NOOP, if error checks are disabled.
  return;
#endif
}

bool NCCLComm::isInitialized() const {
  LockType lock(mutex_);
  return initialized_;
}

bool NCCLComm::isAborted() const {
  LockType lock(mutex_);
  return aborted_;
}

uint64_t NCCLComm::getCommSplitCounter() const {
  return ncclCommSplitCounter_;
}

ncclResult_t NCCLComm::checkForNcclError() {
  LockType lock(mutex_);
#ifdef ENABLE_NCCL_ERROR_CHECKING
  if (ncclAsyncErr_ != ncclSuccess) {
    return ncclAsyncErr_;
  }
  C10D_NCCL_CHECK(
      ncclCommGetAsyncError(ncclComm_, &ncclAsyncErr_), commFailureReason_);
  return ncclAsyncErr_;
#else
  // Always return success, if error checks are disabled.
  return ncclSuccess;
#endif
}

ncclResult_t NCCLComm::registerSegment(void* ptr, size_t size) {
  LockType lock(mutex_);
#ifdef NCCL_HAS_COMM_REGISTER
  // We register only segments from cache allocator
  // which are guaranteed to be with disjoint addr ranges. Thus, a ptr always
  // maps to a unique handle and should not be registered before the current
  // ptr is deregistered and freed.
  TORCH_CHECK(
      registeredSegmentHandles_.count(ptr) == 0,
      "Segment with ptr ",
      ptr,
      " has already been registered on ncclComm_ ",
      ncclComm_);

  void* handle = nullptr;
  // Use getNcclComm to make sure comm is ready before calling nccl APIs
  auto comm = getNcclComm();
  C10D_NCCL_CHECK(
      ncclCommRegister(comm, ptr, size, &handle),
      c10::str(
          "Failed to register segment with ptr ",
          ptr,
          ", size ",
          size,
          " on ncclComm_ ",
          comm));
  registeredSegmentHandles_[ptr] = handle;
  return ncclSuccess;
#else
  return ncclInvalidUsage;
#endif
}

ncclResult_t NCCLComm::deregisterSegment(void* ptr) {
  LockType lock(mutex_);
#ifdef NCCL_HAS_COMM_REGISTER
  TORCH_CHECK(
      registeredSegmentHandles_.count(ptr) == 1,
      "Segment with ptr ",
      ptr,
      " is not registered on ncclComm_ ",
      ncclComm_);

  void* handle = registeredSegmentHandles_[ptr];
  // Use getNcclComm to make sure comm is ready before calling nccl APIs
  auto comm = getNcclComm();
  C10D_NCCL_CHECK(
      ncclCommDeregister(comm, handle),
      c10::str(
          "Failed to deregister segment handle ",
          handle,
          ", with ptr ",
          ptr,
          " on ncclComm_ ",
          comm));
  registeredSegmentHandles_.erase(ptr);
  return ncclSuccess;
#else
  return ncclInvalidUsage;
#endif
}

std::string NCCLComm::repr() const {
  return c10::str((void*)ncclComm_);
}

#if defined(IS_NCCLX) && defined(NCCL_COMM_DUMP)
std::unordered_map<std::string, std::string> NCCLComm::ncclCommDump() {
  std::unordered_map<std::string, std::string> dump;
  if (isAborted()) {
    LOG(INFO) << "Communicator was aborted before trying to dump its state.";
    return dump;
  }
  C10D_NCCL_CHECK(::ncclCommDump(ncclComm_, dump), std::nullopt);
  return dump;
}
#endif

std::string getNcclVersion() {
  static c10::once_flag ncclGetVersionFlag;
  static std::string versionString;

  c10::call_once(ncclGetVersionFlag, []() {
    int version = 0;
    ncclResult_t status = ncclGetVersion(&version);
    // can't compute the version if call did not return successfully or version
    // code < 100 (corresponding to 0.1.0)
    if (status != ncclSuccess || version < 100) {
      versionString = "Unknown NCCL version";
    } else {
      // NCCL changed version coding starting 2.9
      const int majorBase = version < 2900 ? 1000 : 10000;
      const int minorBase = 100;
      auto ncclMajor = version / majorBase;
      auto ncclMinor = (version % majorBase) / minorBase;
      auto ncclPatch =
          version % (ncclMajor * majorBase + ncclMinor * minorBase);
      versionString = std::to_string(ncclMajor) + "." +
          std::to_string(ncclMinor) + "." + std::to_string(ncclPatch);
#ifdef NCCL_SUFFIX
      const auto ncclSuffix = std::string(NCCL_SUFFIX);
      if (!ncclSuffix.empty()) {
        versionString += "." + ncclSuffix;
      }
#endif
    }
  });

  return versionString;
}

#ifdef USE_C10D_NCCL
size_t hashTensors(const std::vector<at::Tensor>& tensors) {
  size_t hash = 0;
  for (auto& tensor : tensors) {
    if (tensor.numel() > 0 && tensor.storage()) {
      size_t data_size = tensor.storage().nbytes();
      if (data_size > 0 && tensor.storage().data_ptr()) {
        auto src = static_cast<const char*>(tensor.storage().data_ptr().get());
        std::vector<char> dst(data_size);
        // This is needed so that we trigger a device synchronization so we can
        // get the collective finished if launched on GPU and hash its output.
        cudaMemcpy(dst.data(), src, data_size, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < data_size; ++i) {
          // Update the hash for each byte in the tensor
          hash = c10::hash_combine(hash, c10::get_hash(dst[i], data_size));
        }
      }
    }
  }
  return hash;
}
#endif

// Default value: 30 minutes
int nccl_nonblocking_timeout() {
  static int timeout = -2; // -2 means not initialized
  if (timeout == -2) {
    const auto val = c10::utils::get_env("TORCH_NCCL_NONBLOCKING_TIMEOUT");
    if (val.has_value() && !val.value().empty()) {
      timeout = stoi(val.value());
    } else {
      // Default value consistent with kBackendDefaultTimeout
      timeout = 30 * 60;
    }
  }
  return timeout;
}

std::string ncclGetErrorWithVersion(ncclResult_t error) {
  return std::string(ncclGetErrorString(error)) + ", NCCL version " +
      getNcclVersion();
}

// Provides additional detail into NCCL error codes based on when these are
// thrown in the NCCL codebase.
std::string getNcclErrorDetailStr(
    ncclResult_t error,
    std::optional<std::string> processGroupFailureReason /* = std::nullopt */
) {
  // Prioritize failure reason provided by PG NCCL first, as it can abort
  // communicators when it encounters collective timeouts, etc.
  if (processGroupFailureReason != std::nullopt) {
    return *processGroupFailureReason;
  }
  std::string interpret;
  std::string err;
#ifdef ENABLE_NCCL_GET_LAST_ERROR
  auto ret = ncclGetLastError(nullptr);
  if (ret) {
    err = "\nLast error:\n" + std::string(ret);
  } else {
    err = "\nLast error: Unknown NCCL Error\n";
  }
#endif
  switch (error) {
    case ncclUnhandledCudaError:
      interpret = "ncclUnhandledCudaError: Call to CUDA function failed.";
      break;
    case ncclSystemError:
      interpret =
          "ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error. ";
#ifndef NCCL_REMOTE_ERROR
      // Before ncclRemoteError was created, unexpected remote disconnect was
      // categorized as ncclSystemError
      interpret += "It can be also caused by unexpected exit of a remote peer.";
#endif
      break;
    case ncclInternalError:
      interpret = "ncclInternalError: Internal check failed.";
      break;
    case ncclInvalidArgument:
      interpret = "ncclInvalidArgument: Invalid value for an argument.";
      break;
    case ncclInvalidUsage:
      interpret =
          "ncclInvalidUsage: This usually reflects invalid usage of NCCL library.";
      break;
#ifdef NCCL_REMOTE_ERROR
    case ncclRemoteError:
      interpret =
          "ncclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.";
      break;
#endif
    default:
      interpret = "Unknown NCCL error!";
  }
  return interpret + err;
}

} // namespace c10d

#endif // USE_C10D_NCCL
