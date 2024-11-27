#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>

#include <c10/util/env.h>

#ifdef USE_C10D_NCCL
#include <mutex>
#include <vector>

namespace c10d {

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
    // If timeout is reached, throw an exception.
    C10D_NCCL_CHECK_TIMEOUT_SLEEP(ncclInProgress, ncclComm_, std::nullopt);
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
