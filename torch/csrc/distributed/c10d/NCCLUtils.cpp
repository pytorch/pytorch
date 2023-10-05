#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>

#include <c10/util/CallOnce.h>
#include <c10/util/env.h>

#ifdef USE_C10D_NCCL

#include <mutex>

namespace c10d {

ncclComm_t NCCLComm::getNcclComm() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (aborted_) {
    auto commFailureMsg = commFailureReason_ != c10::nullopt
        ? c10::str(" Original reason for failure was: ", *commFailureReason_)
        : "";
    TORCH_CHECK(
        false,
        c10::str(
            "NCCL communicator was aborted on rank ",
            rank_,
            ". ",
            commFailureMsg));
  }
  return ncclComm_;
}

std::string getNcclVersion() {
  static c10::once_flag ncclGetVersionFlag;
  static std::string versionString;

  c10::call_once(ncclGetVersionFlag, []() {
    int version;
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
    }
  });

  return versionString;
}

bool nccl_use_nonblocking() {
  static bool nccl_use_nonblocking_ =
      c10::utils::check_env("TORCH_NCCL_USE_COMM_NONBLOCKING") == true;
  if (nccl_use_nonblocking_) {
    TORCH_WARN("Using experimental non-blocking NCCL communicator.");
  }
  return nccl_use_nonblocking_;
}

int _parse_nccl_nonblocking_timeout() {
  const char* val = getenv("TORCH_NCCL_NONBLOCKING_TIMEOUT");
  int timeout = -1;
  if (val) {
    const std::string config(val);
    timeout = std::stoi(config);
    if (!nccl_use_nonblocking() && timeout > 0) {
      TORCH_WARN(
          "TORCH_NCCL_NONBLOCKING_TIMEOUT has no effect when TORCH_NCCL_USE_COMM_NONBLOCKING is false.");
      timeout = -1;
    }
  }
  return timeout;
}

int nccl_nonblocking_timeout() {
  static int timeout = _parse_nccl_nonblocking_timeout();
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
    c10::optional<std::string> processGroupFailureReason /* = c10::nullopt */
) {
  // Prioritize failure reason provided by PG NCCL first, as it can abort
  // communicators when it encounters collective timeouts, etc.
  if (processGroupFailureReason != c10::nullopt) {
    return *processGroupFailureReason;
  }
  std::string interpret;
  std::string err;
#ifdef ENABLE_NCCL_GET_LAST_ERROR
  err = "\nLast error:\n" + std::string(ncclGetLastError(NULL));
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
