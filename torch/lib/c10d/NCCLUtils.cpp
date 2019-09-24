#include <c10d/NCCLUtils.hpp>

namespace c10d {
std::string getNcclVersion() {
  int version;
  ncclResult_t status = ncclGetVersion(&version);
  if (status != ncclSuccess) {
    return "Unknown NCCL version";
  }
  auto ncclMajor = version / 1000;
  auto ncclMinor = (version % 1000) / 100;
  auto ncclPatch = version % (ncclMajor * 1000 + ncclMinor * 100);
  return std::to_string(ncclMajor) + "." + std::to_string(ncclMinor) + "." +
      std::to_string(ncclPatch);
}

std::string ncclGetErrorWithVersion(ncclResult_t error) {
  return std::string(ncclGetErrorString(error)) + ", NCCL version " +
      getNcclVersion();
}
} // namespace c10d
