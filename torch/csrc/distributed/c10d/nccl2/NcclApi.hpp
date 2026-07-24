// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef USE_C10D_NCCL

#include <mutex>
#include <string>
#include <string_view>

#include <nccl.h>

// NCCL_SHRINK_ABORT was introduced in NCCL 2.27 alongside ncclCommShrink.
// Define a fallback so dependents compile against older NCCL headers; the
// commShrink wrapper in NcclApi.cpp is itself version-guarded and returns
// ncclInvalidUsage on older NCCL, so the value is never read at runtime.
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 27, 0) && !defined(NCCL_SHRINK_ABORT)
#define NCCL_SHRINK_ABORT 0x01
#endif

// ncclCommSuspend/ncclCommResume/ncclCommMemStats (memory offload) landed in
// NCCL 2.29.7. NCCL_SUSPEND_MEM is the only suspend flag; define a fallback so
// callers compile against older headers -- the wrappers return ncclInvalidUsage
// there, so the value is never read at runtime.
#ifndef NCCL_SUSPEND_MEM
#define NCCL_SUSPEND_MEM 0x01
#endif

// ncclWindow_t was introduced in NCCL 2.27; the window/RMA APIs
// (ncclCommWindowRegister, ncclWinGetUserPtr, ncclPutSignal, ncclSignal,
// ncclWaitSignal) only landed in 2.29+. Provide an opaque alias on older
// headers so the NcclApi interface can be declared unconditionally -- the
// wrapper implementations return ncclInvalidUsage on older NCCL and
// TorchCommNCCLWindow throws at construction.
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 27, 0)
typedef struct ncclWindow_vidmem* ncclWindow_t;
#endif

#ifndef NCCL_WIN_COLL_SYMMETRIC
#define NCCL_WIN_COLL_SYMMETRIC 0x01
#endif

namespace c10d::nccl2 {
/**
 * Abstract interface for NCCL API operations.
 * This allows for dependency injection and testing by providing
 * a way to override NCCL API calls.
 */
class NcclApi {
 public:
  virtual ~NcclApi() = default;

  // Error handling
  virtual std::string_view getErrorString(ncclResult_t result) = 0;
  virtual std::string getLastError(ncclComm_t comm) = 0;

  // Unique ID generation
  [[nodiscard]] virtual ncclResult_t getUniqueId(ncclUniqueId* uniqueId) = 0;

  // Communicator management
  [[nodiscard]] virtual ncclResult_t commInitRankConfig(
      ncclComm_t* comm,
      int nranks,
      ncclUniqueId commId,
      int rank,
      ncclConfig_t* config) = 0;

  [[nodiscard]] virtual ncclResult_t commDestroy(ncclComm_t comm) = 0;

  [[nodiscard]] virtual ncclResult_t commAbort(ncclComm_t comm) = 0;

  [[nodiscard]] virtual ncclResult_t commRevoke(ncclComm_t comm) = 0;

  [[nodiscard]] virtual ncclResult_t commGetAsyncError(
      ncclComm_t comm,
      ncclResult_t* asyncError) = 0;

  [[nodiscard]] virtual ncclResult_t commSplit(
      ncclComm_t comm,
      int color,
      int key,
      ncclComm_t* newcomm,
      ncclConfig_t* config) = 0;

  [[nodiscard]] virtual ncclResult_t commShrink(
      ncclComm_t comm,
      int* excludeRanksList,
      int excludeRanksCount,
      ncclComm_t* newcomm,
      ncclConfig_t* config,
      int shrinkFlags) = 0;

  [[nodiscard]] virtual ncclResult_t commGetUniqueId(
      ncclComm_t comm,
      ncclUniqueId* uniqueId) = 0;

  [[nodiscard]] virtual ncclResult_t commGrow(
      ncclComm_t comm,
      int nRanks,
      const ncclUniqueId* uniqueId,
      int rank,
      ncclComm_t* newcomm,
      ncclConfig_t* config) = 0;

  // Memory registration
  [[nodiscard]] virtual ncclResult_t commRegister(
      ncclComm_t comm,
      void* buffer,
      size_t size,
      void** handle) = 0;

  [[nodiscard]] virtual ncclResult_t commDeregister(
      ncclComm_t comm,
      void* handle) = 0;

  // Point-to-point operations
  [[nodiscard]] virtual ncclResult_t send(
      const void* sendbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t recv(
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  // Collective operations
  [[nodiscard]] virtual ncclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t bcast(
      void* buff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t allReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t reduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      int root,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t allGather(
      const void* sendbuff,
      void* recvbuff,
      size_t sendcount,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      size_t recvcount,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t allToAll(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  // Group operations
  [[nodiscard]] virtual ncclResult_t groupStart() = 0;
  [[nodiscard]] virtual ncclResult_t groupEnd() = 0;

  [[nodiscard]] virtual ncclResult_t commUserRank(
      const ncclComm_t comm,
      int* userRank) = 0;
  [[nodiscard]] virtual ncclResult_t commCount(
      const ncclComm_t comm,
      int* count) = 0;

  [[nodiscard]] virtual ncclResult_t redOpCreatePreMulSum(
      ncclRedOp_t* op,
      void* scalar,
      ncclDataType_t datatype,
      ncclScalarResidence_t residence,
      ncclComm_t comm) = 0;
  [[nodiscard]] virtual ncclResult_t redOpDestroy(
      ncclRedOp_t op,
      ncclComm_t comm) = 0;

  [[nodiscard]] virtual ncclResult_t memAlloc(void** buff, size_t size) = 0;
  [[nodiscard]] virtual ncclResult_t memFree(void* buff) = 0;

  // Memory offload (suspend/resume) operations.
  // Available on NCCL 2.29.7+ (older NCCL returns ncclInvalidUsage).
  [[nodiscard]] virtual ncclResult_t commSuspend(
      ncclComm_t comm,
      int flags) = 0;
  [[nodiscard]] virtual ncclResult_t commResume(ncclComm_t comm) = 0;
  [[nodiscard]] virtual ncclResult_t commMemStats(
      ncclComm_t comm,
      int stat,
      uint64_t* value) = 0;

  // Window / one-sided RMA operations.
  // Available on NCCL 2.29+ (older NCCL returns ncclInvalidUsage).
  [[nodiscard]] virtual ncclResult_t commWindowRegister(
      ncclComm_t comm,
      void* buffer,
      size_t size,
      ncclWindow_t* win,
      int winFlags) = 0;

  [[nodiscard]] virtual ncclResult_t commWindowDeregister(
      ncclComm_t comm,
      ncclWindow_t win) = 0;

  [[nodiscard]] virtual ncclResult_t winGetUserPtr(
      ncclComm_t comm,
      ncclWindow_t win,
      void** outUserPtr) = 0;

  [[nodiscard]] virtual ncclResult_t putSignal(
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
      cudaStream_t stream) = 0;

  [[nodiscard]] virtual ncclResult_t signal(
      int peer,
      int sigIdx,
      int ctx,
      unsigned int flags,
      ncclComm_t comm,
      cudaStream_t stream) = 0;

  // waitSignal takes a single descriptor (peer, sigIdx, ctx, opCnt) -- the only
  // shape currently consumed by TorchCommNCCLWindow. Multi-descriptor waits can
  // be added if needed.
  [[nodiscard]] virtual ncclResult_t waitSignal(
      int peer,
      int sigIdx,
      int ctx,
      int opCnt,
      ncclComm_t comm,
      cudaStream_t stream) = 0;
};

/**
 * Default implementation that calls the underlying NCCL APIs directly.
 */
class DefaultNcclApi : public NcclApi {
 public:
  ~DefaultNcclApi() override = default;

  // Error handling
  std::string_view getErrorString(ncclResult_t result) override;
  std::string getLastError(ncclComm_t comm) override;

  // Unique ID generation
  [[nodiscard]] ncclResult_t getUniqueId(ncclUniqueId* uniqueId) override;

  // Communicator management
  [[nodiscard]] ncclResult_t commInitRankConfig(
      ncclComm_t* comm,
      int nranks,
      ncclUniqueId commId,
      int rank,
      ncclConfig_t* config) override;

  [[nodiscard]] ncclResult_t commDestroy(ncclComm_t comm) override;

  [[nodiscard]] ncclResult_t commAbort(ncclComm_t comm) override;

  [[nodiscard]] ncclResult_t commRevoke(ncclComm_t comm) override;

  [[nodiscard]] ncclResult_t commGetAsyncError(
      ncclComm_t comm,
      ncclResult_t* asyncError) override;

  [[nodiscard]] ncclResult_t commSplit(
      ncclComm_t comm,
      int color,
      int key,
      ncclComm_t* newcomm,
      ncclConfig_t* config) override;

  [[nodiscard]] ncclResult_t commShrink(
      ncclComm_t comm,
      int* excludeRanksList,
      int excludeRanksCount,
      ncclComm_t* newcomm,
      ncclConfig_t* config,
      int shrinkFlags) override;

  [[nodiscard]] ncclResult_t commGetUniqueId(
      ncclComm_t comm,
      ncclUniqueId* uniqueId) override;

  [[nodiscard]] ncclResult_t commGrow(
      ncclComm_t comm,
      int nRanks,
      const ncclUniqueId* uniqueId,
      int rank,
      ncclComm_t* newcomm,
      ncclConfig_t* config) override;

  [[nodiscard]] ncclResult_t commRegister(
      ncclComm_t comm,
      void* buffer,
      size_t size,
      void** handle) override;

  [[nodiscard]] ncclResult_t commDeregister(ncclComm_t comm, void* handle)
      override;

  [[nodiscard]] ncclResult_t commSuspend(ncclComm_t comm, int flags) override;
  [[nodiscard]] ncclResult_t commResume(ncclComm_t comm) override;
  [[nodiscard]] ncclResult_t commMemStats(
      ncclComm_t comm,
      int stat,
      uint64_t* value) override;

  // Point-to-point operations
  [[nodiscard]] ncclResult_t send(
      const void* sendbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      cudaStream_t stream) override;

  [[nodiscard]] ncclResult_t recv(
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int peer,
      ncclComm_t comm,
      cudaStream_t stream) override;

  // Collective operations
  [[nodiscard]] ncclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      cudaStream_t stream) override;

  [[nodiscard]] ncclResult_t bcast(
      void* buff,
      size_t count,
      ncclDataType_t datatype,
      int root,
      ncclComm_t comm,
      cudaStream_t stream) override;

  [[nodiscard]] ncclResult_t allReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      cudaStream_t stream) override;

  [[nodiscard]] ncclResult_t reduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      int root,
      ncclComm_t comm,
      cudaStream_t stream) override;

  [[nodiscard]] ncclResult_t allGather(
      const void* sendbuff,
      void* recvbuff,
      size_t sendcount,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) override;

  [[nodiscard]] ncclResult_t reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      size_t recvcount,
      ncclDataType_t datatype,
      ncclRedOp_t op,
      ncclComm_t comm,
      cudaStream_t stream) override;

  [[nodiscard]] ncclResult_t allToAll(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      ncclDataType_t datatype,
      ncclComm_t comm,
      cudaStream_t stream) override;

  // Group operations
  [[nodiscard]] ncclResult_t groupStart() override;
  [[nodiscard]] ncclResult_t groupEnd() override;

  [[nodiscard]] ncclResult_t commUserRank(const ncclComm_t comm, int* userRank)
      override;
  [[nodiscard]] ncclResult_t commCount(const ncclComm_t comm, int* count)
      override;

  [[nodiscard]] ncclResult_t redOpCreatePreMulSum(
      ncclRedOp_t* op,
      void* scalar,
      ncclDataType_t datatype,
      ncclScalarResidence_t residence,
      ncclComm_t comm) override;
  [[nodiscard]] ncclResult_t redOpDestroy(ncclRedOp_t op, ncclComm_t comm)
      override;

  [[nodiscard]] ncclResult_t memAlloc(void** buff, size_t size) override;
  [[nodiscard]] ncclResult_t memFree(void* buff) override;

  [[nodiscard]] ncclResult_t commWindowRegister(
      ncclComm_t comm,
      void* buffer,
      size_t size,
      ncclWindow_t* win,
      int winFlags) override;

  [[nodiscard]] ncclResult_t commWindowDeregister(
      ncclComm_t comm,
      ncclWindow_t win) override;

  [[nodiscard]] ncclResult_t winGetUserPtr(
      ncclComm_t comm,
      ncclWindow_t win,
      void** outUserPtr) override;

  [[nodiscard]] ncclResult_t putSignal(
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
      cudaStream_t stream) override;

  [[nodiscard]] ncclResult_t signal(
      int peer,
      int sigIdx,
      int ctx,
      unsigned int flags,
      ncclComm_t comm,
      cudaStream_t stream) override;

  [[nodiscard]] ncclResult_t waitSignal(
      int peer,
      int sigIdx,
      int ctx,
      int opCnt,
      ncclComm_t comm,
      cudaStream_t stream) override;

 private:
  mutable std::mutex api_mutex_;
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
