#include <c10d/NCCLUtils.hpp>
#include <torch/csrc/cuda/nccl.h>

#ifdef USE_C10D_NCCL

#include <mutex>

namespace c10d {

std::string getNcclVersion() {
  static std::once_flag ncclGetVersionFlag;
  static std::string versionString;

  std::call_once(ncclGetVersionFlag, []() {
    int version = torch::cuda::nccl::version();
    if (version <= 0) {
      versionString = "Unknown NCCL version";
    } else {
      // Ref: https://github.com/NVIDIA/nccl/blob/v2.10.3-1/src/nccl.h.in#L22
      int majorDiv = (version >= 10000)? 10000 : 1000;
      auto ncclMajor = version / majorDiv;
      auto ncclMinor = (version % majorDiv) / 100;
      auto ncclPatch = version % (ncclMajor * majorDiv + ncclMinor * 100);
      versionString = std::to_string(ncclMajor) + "." +
          std::to_string(ncclMinor) + "." + std::to_string(ncclPatch);
    }
  });

  return versionString;
}

std::string ncclGetErrorWithVersion(ncclResult_t error) {
  return std::string(ncclGetErrorString(error)) + ", NCCL version " +
      getNcclVersion();
}

} // namespace c10d

#endif // USE_C10D_NCCL
