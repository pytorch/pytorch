#include <c10d/NCCLUtils.hpp>
#include <mutex>

namespace c10d {

std::string getNcclVersion() {
  static std::once_flag ncclGetVersionFlag;
  static std::string versionString;

  std::call_once(ncclGetVersionFlag, []() {
    int version;
    ncclResult_t status = ncclGetVersion(&version);
    // can't compute the version if call did not return successfully or version
    // code < 100 (corresponding to 0.1.0)
    if (status != ncclSuccess || version < 100) {
      versionString = "Unknown NCCL version";
    } else {
      auto ncclMajor = version / 1000;
      auto ncclMinor = (version % 1000) / 100;
      auto ncclPatch = version % (ncclMajor * 1000 + ncclMinor * 100);
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

#ifdef ENABLE_NCCL_P2P_SUPPORT
ncclResult_t gpuAlltoall(
    void* sendbuff,
    void* recvbuff,
    size_t count,
    size_t elem_size,
    ncclDataType_t type,
    ncclComm_t comm,
    cudaStream_t stream) {
  int numRanks;
  int myRank;
  C10D_NCCL_CHECK(ncclCommCount(comm, &numRanks));
  C10D_NCCL_CHECK(ncclCommUserRank(comm, &myRank));
  size_t rankOffset = count * elem_size;
  for (int r = 0; r < numRanks; r++) {
    int sendRank = (myRank + r) % numRanks;
    int recvRank = (myRank - r + numRanks) % numRanks;
    C10D_NCCL_CHECK(ncclSend(
        ((char*)sendbuff) + sendRank * rankOffset,
        count,
        type,
        sendRank,
        comm,
        stream));
    C10D_NCCL_CHECK(ncclRecv(
        ((char*)recvbuff) + recvRank * rankOffset,
        count,
        type,
        recvRank,
        comm,
        stream));
  }
  return ncclSuccess;
}

ncclResult_t gpuAlltoallv(
    void* sendbuff,
    const int* sendcounts,
    const int* sdispls,
    void* recvbuff,
    const int* recvcounts,
    const int* rdispls,
    size_t elem_size,
    ncclDataType_t type,
    ncclComm_t comm,
    cudaStream_t stream) {
  int numRanks;
  int myRank;
  C10D_NCCL_CHECK(ncclCommCount(comm, &numRanks));
  C10D_NCCL_CHECK(ncclCommUserRank(comm, &myRank));
  for (int r = 0; r < numRanks; r++) {
    int sendRank = (myRank + r) % numRanks;
    int recvRank = (myRank - r + numRanks) % numRanks;
    C10D_NCCL_CHECK(ncclSend(
        ((char*)sendbuff) + sdispls[sendRank] * elem_size,
        sendcounts[sendRank],
        type,
        sendRank,
        comm,
        stream));
    C10D_NCCL_CHECK(ncclRecv(
        ((char*)recvbuff) + rdispls[recvRank] * elem_size,
        recvcounts[recvRank],
        type,
        recvRank,
        comm,
        stream));
  }
  return ncclSuccess;
}
#endif

} // namespace c10d
