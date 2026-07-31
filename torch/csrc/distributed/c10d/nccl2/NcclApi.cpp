// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <fmt/core.h>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/NcclApi.hpp>
#include <string_view>
#include <tuple>

namespace c10d::nccl2 {

// DefaultNcclApi implementation
std::string_view DefaultNcclApi::getErrorString(ncclResult_t result) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGetErrorString(result);
}

std::string DefaultNcclApi::getLastError(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0)
  const char* lastError = ncclGetLastError(comm);
  return lastError ? std::string(lastError) : std::string();
#else
  std::ignore = comm;
  return std::string();
#endif
}

ncclResult_t DefaultNcclApi::getUniqueId(ncclUniqueId* uniqueId) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGetUniqueId(uniqueId);
}

ncclResult_t DefaultNcclApi::commInitRankConfig(
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId commId,
    int rank,
    ncclConfig_t* config) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommInitRankConfig(comm, nranks, commId, rank, config);
}

ncclResult_t DefaultNcclApi::commDestroy(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommDestroy(comm);
}

ncclResult_t DefaultNcclApi::commAbort(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommAbort(comm);
}

ncclResult_t DefaultNcclApi::commRevoke(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
// RCCL advertises NCCL_VERSION_CODE >= 2.28 but does not provide
// ncclCommRevoke; on ROCm fall through to the unsupported path.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0) && !defined(USE_ROCM)
  return ncclCommRevoke(comm, 0);
#else
  std::ignore = comm;
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclCommRevoke API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::commGetAsyncError(
    ncclComm_t comm,
    ncclResult_t* asyncError) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommGetAsyncError(comm, asyncError);
}

ncclResult_t DefaultNcclApi::commSplit(
    ncclComm_t comm,
    int color,
    int key,
    ncclComm_t* newcomm,
    ncclConfig_t* config) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommSplit(comm, color, key, newcomm, config);
}

ncclResult_t DefaultNcclApi::commShrink(
    ncclComm_t comm,
    int* excludeRanksList,
    int excludeRanksCount,
    ncclComm_t* newcomm,
    ncclConfig_t* config,
    int shrinkFlags) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  return ncclCommShrink(
      comm, excludeRanksList, excludeRanksCount, newcomm, config, shrinkFlags);
#else
  std::ignore = std::tie(
      comm, excludeRanksList, excludeRanksCount, newcomm, config, shrinkFlags);
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclCommShrink API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::commGetUniqueId(
    ncclComm_t comm,
    ncclUniqueId* uniqueId) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  return ncclCommGetUniqueId(comm, uniqueId);
#else
  std::ignore = std::tie(comm, uniqueId);
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclCommGetUniqueId API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::commGrow(
    ncclComm_t comm,
    int nRanks,
    const ncclUniqueId* uniqueId,
    int rank,
    ncclComm_t* newcomm,
    ncclConfig_t* config) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  return ncclCommGrow(comm, nRanks, uniqueId, rank, newcomm, config);
#else
  std::ignore = std::tie(comm, nRanks, uniqueId, rank, newcomm, config);
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclCommGrow API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::commRegister(
    ncclComm_t comm,
    void* buffer,
    size_t size,
    void** handle) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
  return ncclCommRegister(comm, buffer, size, handle);
#else
  throw std::runtime_error(fmt::format(
      "NCCL version {} does not support ncclCommRegister API",
      NCCL_VERSION_CODE));
#endif
}

ncclResult_t DefaultNcclApi::commDeregister(ncclComm_t comm, void* handle) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
  return ncclCommDeregister(comm, handle);
#else
  throw std::runtime_error(fmt::format(
      "NCCL version {} does not support ncclCommDeregister API",
      NCCL_VERSION_CODE));
#endif
}

ncclResult_t DefaultNcclApi::send(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclSend(sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t DefaultNcclApi::recv(
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}

ncclResult_t DefaultNcclApi::broadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t DefaultNcclApi::bcast(
    void* buff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclBcast(buff, count, datatype, root, comm, stream);
}

ncclResult_t DefaultNcclApi::allReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t DefaultNcclApi::reduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    int root,
    ncclComm_t comm,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclReduce(
      sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t DefaultNcclApi::allGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

ncclResult_t DefaultNcclApi::reduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclReduceScatter(
      sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

ncclResult_t DefaultNcclApi::allToAll(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  return ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
#else
  std::ignore = std::tie(sendbuff, recvbuff, count, datatype, comm, stream);
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclAlltoAll API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::groupStart() {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGroupStart();
}

ncclResult_t DefaultNcclApi::groupEnd() {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGroupEnd();
}

ncclResult_t DefaultNcclApi::commUserRank(const ncclComm_t comm, int* myRank) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommUserRank(comm, myRank);
}

ncclResult_t DefaultNcclApi::commCount(const ncclComm_t comm, int* count) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclCommCount(comm, count);
}

ncclResult_t DefaultNcclApi::redOpCreatePreMulSum(
    ncclRedOp_t* op,
    void* scalar,
    ncclDataType_t datatype,
    ncclScalarResidence_t residence,
    ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
}

ncclResult_t DefaultNcclApi::redOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclRedOpDestroy(op, comm);
}

ncclResult_t DefaultNcclApi::memAlloc(void** buff, size_t size) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
  return ncclMemAlloc(buff, size);
#else
  throw std::runtime_error(fmt::format(
      "NCCL version {} does not support ncclMemAlloc API", NCCL_VERSION_CODE));
#endif
}

ncclResult_t DefaultNcclApi::memFree(void* buff) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
  return ncclMemFree(buff);
#else
  throw std::runtime_error(fmt::format(
      "NCCL version {} does not support ncclMemFree API", NCCL_VERSION_CODE));
#endif
}

ncclResult_t DefaultNcclApi::commSuspend(ncclComm_t comm, int flags) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
  return ncclCommSuspend(comm, flags);
#else
  std::ignore = std::tie(comm, flags);
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclCommSuspend API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::commResume(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
  return ncclCommResume(comm);
#else
  std::ignore = comm;
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclCommResume API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::commMemStats(
    ncclComm_t comm,
    int stat,
    uint64_t* value) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
  return ncclCommMemStats(comm, static_cast<ncclCommMemStat_t>(stat), value);
#else
  std::ignore = std::tie(comm, stat, value);
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclCommMemStats API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::commWindowRegister(
    ncclComm_t comm,
    void* buffer,
    size_t size,
    ncclWindow_t* win,
    int winFlags) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  return ncclCommWindowRegister(comm, buffer, size, win, winFlags);
#else
  std::ignore = std::tie(comm, buffer, size, win, winFlags);
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclCommWindowRegister API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::commWindowDeregister(
    ncclComm_t comm,
    ncclWindow_t win) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  return ncclCommWindowDeregister(comm, win);
#else
  std::ignore = std::tie(comm, win);
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclCommWindowDeregister API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::winGetUserPtr(
    ncclComm_t comm,
    ncclWindow_t win,
    void** outUserPtr) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  return ncclWinGetUserPtr(comm, win, outUserPtr);
#else
  std::ignore = std::tie(comm, win, outUserPtr);
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclWinGetUserPtr API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::putSignal(
    const void* localbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclWindow_t peerWin,
    size_t peerWinOffset,
    int sigIdx,
    int ctx,
    unsigned int flags,
    ncclComm_t comm,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  return ncclPutSignal(
      localbuff,
      count,
      datatype,
      peer,
      peerWin,
      peerWinOffset,
      sigIdx,
      ctx,
      flags,
      comm,
      stream);
#else
  std::ignore = std::tie(
      localbuff,
      count,
      datatype,
      peer,
      peerWin,
      peerWinOffset,
      sigIdx,
      ctx,
      flags,
      comm,
      stream);
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclPutSignal API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::signal(
    int peer,
    int sigIdx,
    int ctx,
    unsigned int flags,
    ncclComm_t comm,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  return ncclSignal(peer, sigIdx, ctx, flags, comm, stream);
#else
  std::ignore = std::tie(peer, sigIdx, ctx, flags, comm, stream);
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclSignal API";
  return ncclInvalidUsage;
#endif
}

ncclResult_t DefaultNcclApi::waitSignal(
    int peer,
    int sigIdx,
    int ctx,
    int opCnt,
    ncclComm_t comm,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  ncclWaitSignalDesc_t desc;
  desc.opCnt = opCnt;
  desc.peer = peer;
  desc.sigIdx = sigIdx;
  desc.ctx = ctx;
  return ncclWaitSignal(1, &desc, comm, stream);
#else
  std::ignore = std::tie(peer, sigIdx, ctx, opCnt, comm, stream);
  TC_LOG(ERROR) << "NCCL version " << NCCL_VERSION_CODE
                << " does not support ncclWaitSignal API";
  return ncclInvalidUsage;
#endif
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
