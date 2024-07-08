#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

#include <c10/util/CallOnce.h>
#include <c10/util/env.h>
#include <algorithm>

#ifdef USE_C10D_NCCL
#include <vector>

#include <cuda_runtime.h>
#include <mutex>

namespace {
constexpr int64_t kCommInitBusyWaitMillis = 10;
} // namespace

namespace c10d {

ncclComm_t NCCLComm::getNcclComm() {
  std::unique_lock<std::mutex> lock(mutex_);
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
  // only wait for initialization if nonblocking mode is enabled
  if (!initialized_ && nccl_use_nonblocking()) {
    waitUntilInitialized(nccl_nonblocking_timeout());
  }

  return ncclComm_;
}

void NCCLComm::waitUntilInitialized(int timeoutSecs) {
  auto startTimepoint = std::chrono::steady_clock::now();
  while (!initialized_) {
    if (ncclComm_) {
      ncclResult_t result;
      ncclCommGetAsyncError(ncclComm_, &result);
      if (result == ncclSuccess) {
        LOG(INFO) << "Rank " << rank_ << ": NCCL communicator is initialized.";
        initialized_ = true;
        break;
      }
    }
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           currentTimepoint - startTimepoint)
                           .count();
    if (timeElapsed > timeoutSecs) {
      std::string err = "NCCL timeout in communicator initialization.";
      TORCH_CHECK_WITH(DistBackendError, false, err);
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kCommInitBusyWaitMillis));
  }
}

#if defined(NCCL_HAS_COMM_SPLIT) && !defined(FBCODE_CAFFE2)
// last argument to split() API is not used to support
// multiple implementations
std::shared_ptr<NCCLComm> NCCLComm::split(
    NCCLComm* source,
    int color_id,
    int rank,
    ncclConfig_t& config,
    std::vector<uint64_t>& ranks_ull) {
  auto comm = std::make_shared<NCCLComm>();
  C10D_NCCL_CHECK(
      ncclCommSplit(
          source->ncclComm_, color_id, rank, &(comm->ncclComm_), &config),
      std::nullopt);
  ++source->ncclCommSplitCounter_;
  comm->rank_ = rank;
  return comm;
}
#endif

#ifndef FBCODE_CAFFE2
bool shouldBroadcastNCCLUniqueID(bool isSendRecvSelf) {
  // For point-to-point communication on the same process, don't need broadcast.
  return !isSendRecvSelf;
}
#endif

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
#ifdef NCCL_SUFFIX
      const auto ncclSuffix = std::string(NCCL_SUFFIX);
      if (ncclSuffix.length()) {
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
        char* dst = (char*)std::calloc(data_size, sizeof(char));
        // This is needed so that we trigger a device synchronization so we can
        // get the collective finished if launched on GPU and hash its output.
        cudaMemcpy(dst, src, data_size, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < data_size; ++i) {
          // Update the hash for each byte in the tensor
          hash = c10::hash_combine(
              hash, c10::get_hash(((char*)dst)[i], data_size));
        }
        free(dst);
      }
    }
  }
  return hash;
}
#endif

bool nccl_use_nonblocking() {
  static bool nccl_use_nonblocking_ =
      c10::utils::check_env("TORCH_NCCL_USE_COMM_NONBLOCKING") == true;
  if (nccl_use_nonblocking_) {
    TORCH_WARN_ONCE("Using experimental non-blocking NCCL communicator.");
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
  auto ret = ncclGetLastError(NULL);
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

control_plane::RegisterHandler dumpHandler{
    "dump_nccl_trace_pickle",
    [](const control_plane::Request& req, control_plane::Response& res) {
      const auto params = req.params();
      size_t validParamCount = 0;

      // valid params
      const std::string includeCollectivesStr = "includecollectives";
      const std::string includeStackTracesStr = "includestacktraces";
      const std::string onlyActiveStr = "onlyactive";

      std::unordered_map<std::string, bool> expectedParams = {
          {includeCollectivesStr, true},
          {includeStackTracesStr, true},
          {onlyActiveStr, false}};

      for (const auto& [paramName, paramValue] : params) {
        auto it = expectedParams.find(paramName);
        if (it != expectedParams.end()) {
          validParamCount++;
          if (paramValue == "true") {
            it->second = true;
          } else if (paramValue == "false") {
            it->second = false;
          } else {
            res.setStatus(400);
            res.setContent(
                "Invalid value for " + paramName +
                    " valid values are true or false",
                "text/plain");
            return;
          }
        }
      }
      if (validParamCount < params.size()) {
        res.setStatus(400);
        res.setContent(
            "Invalid parameters - unexpected param passed in", "text/plain");
        return;
      }
      res.setContent(
          dump_nccl_trace(
              expectedParams[includeCollectivesStr],
              expectedParams[includeStackTracesStr],
              expectedParams[onlyActiveStr]),
          "application/octet-stream");
    }};

control_plane::RegisterHandler jsonDumpHandler{
    "dump_nccl_trace_json",
    [](const control_plane::Request& req, control_plane::Response& res) {
      const auto params = req.params();
      size_t validParamCount = 0;

      // valid params
      const std::string includeCollectivesStr = "includecollectives";
      const std::string onlyActiveStr = "onlyactive";

      std::unordered_map<std::string, bool> expectedParams = {
          {includeCollectivesStr, true}, {onlyActiveStr, false}};

      for (const auto& [paramName, paramValue] : params) {
        auto it = expectedParams.find(paramName);
        if (it != expectedParams.end()) {
          validParamCount++;
          if (paramValue == "true") {
            it->second = true;
          } else if (paramValue == "false") {
            it->second = false;
          } else {
            res.setStatus(400);
            res.setContent(
                "Invalid value for " + paramName +
                    " valid values are true or false",
                "text/plain");
            return;
          }
        }
      }
      if (validParamCount < params.size()) {
        res.setStatus(400);
        res.setContent(
            "Invalid parameters - unexpected param passed in", "text/plain");
        return;
      }
      res.setStatus(200);
      res.setContent(
          dump_nccl_trace_json(
              expectedParams[includeCollectivesStr],
              expectedParams[onlyActiveStr]),
          "application/json");
    }};

void DebugInfoWriter::write(const std::string& ncclTrace) {
  // Open a file for writing. The ios::binary flag is used to write data as
  // binary.
  std::ofstream file(filename_, std::ios::binary);

  // Check if the file was opened successfully.
  if (!file.is_open()) {
    LOG(ERROR) << "Error opening file for writing NCCLPG debug info: "
               << filename_;
    return;
  }

  file.write(ncclTrace.data(), ncclTrace.size());
  LOG(INFO) << "Finished writing NCCLPG debug info to " << filename_;
}

DebugInfoWriter& DebugInfoWriter::getWriter(int rank) {
  if (writer_ == nullptr) {
    std::string fileNamePrefix = getCvarString(
        {"TORCH_NCCL_DEBUG_INFO_TEMP_FILE"}, "/tmp/nccl_trace_rank_");
    // Using std::unique_ptr here to auto-delete the writer object
    // when the pointer itself is destroyed.
    std::unique_ptr<DebugInfoWriter> writerPtr(
        new DebugInfoWriter(fileNamePrefix, rank));
    DebugInfoWriter::registerWriter(std::move(writerPtr));
  }
  return *writer_;
}

void DebugInfoWriter::registerWriter(std::unique_ptr<DebugInfoWriter> writer) {
  TORCH_CHECK_WITH(
      DistBackendError,
      hasWriterRegistered_.load() == false,
      "debugInfoWriter already registered");
  hasWriterRegistered_.store(true);
  writer_ = std::move(writer);
}

std::unique_ptr<DebugInfoWriter> DebugInfoWriter::writer_ = nullptr;
std::atomic<bool> DebugInfoWriter::hasWriterRegistered_(false);

float getDurationFromEvent(
    at::cuda::CUDAEvent& ncclStartEvent,
    at::cuda::CUDAEvent& ncclEndEvent) {
  TORCH_CHECK(
      ncclEndEvent.query(),
      "getDuration can only be called after work is succeeded.")
  return ncclStartEvent.elapsed_time(ncclEndEvent);
}

} // namespace c10d

#endif // USE_C10D_NCCL
