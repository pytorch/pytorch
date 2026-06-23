/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Minimal, self-contained subset of NCCL's profiler plugin ABI (v6), copied
// verbatim from the NCCL profiler plugin headers (err.h / common.h / profiler.h
// / profiler_v5.h / profiler_v6.h) so torch can export an `ncclProfiler_v6`
// symbol without an NCCL header dependency. Only the types our plugin reads /
// implements are included; the struct LAYOUTS are unchanged -- they are the ABI
// contract NCCL passes by value, so do not edit them. See nccl_profiler.cpp.

#pragma once

#include <sys/types.h> // pid_t (proxyOp descriptor)
#include <cstddef>
#include <cstdint>

extern "C" {

// --- err.h ---------------------------------------------------------------
typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6
} ncclResult_t;

// --- common.h ------------------------------------------------------------
typedef enum {
  NCCL_LOG_NONE = 0,
  NCCL_LOG_VERSION = 1,
  NCCL_LOG_WARN = 2,
  NCCL_LOG_INFO = 3,
  NCCL_LOG_ABORT = 4,
  NCCL_LOG_TRACE = 5
} ncclDebugLogLevel;

typedef void (*ncclDebugLogger_t)(
    ncclDebugLogLevel level,
    unsigned long flags,
    const char* file,
    int line,
    const char* fmt,
    ...);

// --- profiler.h: event-type mask -----------------------------------------
enum {
  ncclProfileGroup = (1 << 0),
  ncclProfileColl = (1 << 1), // host collective call event type
  ncclProfileP2p = (1 << 2), // host point-to-point call event type
  ncclProfileProxyOp = (1 << 3),
  ncclProfileProxyStep = (1 << 4),
  ncclProfileProxyCtrl = (1 << 5),
  ncclProfileKernelCh = (1 << 6),
  ncclProfileNetPlugin = (1 << 7),
  ncclProfileGroupApi = (1 << 8),
  ncclProfileCollApi = (1 << 9),
  ncclProfileP2pApi = (1 << 10),
  ncclProfileKernelLaunch = (1 << 11),
  ncclProfileCeColl = (1 << 12),
  ncclProfileCeSync = (1 << 13),
  ncclProfileCeBatch = (1 << 14),
};

// The event-state enum (ncclProfilerEventState_t) is int-sized; we implement
// recordEventState as a no-op and never read the value, so an int alias is ABI-
// equivalent for the vtable signature.
typedef int ncclProfilerEventState_v6_t;

// --- profiler_v5.h: recordEventState args (v6 reuses v5) ------------------
typedef union {
  struct {
    size_t transSize;
  } proxyStep;
  struct {
    int appendedProxyOps;
  } proxyCtrl;
  struct {
    void* data;
  } netPlugin;
  struct {
    uint64_t pTimer;
  } kernelCh;
} ncclProfilerEventStateArgs_v6_t;

// --- profiler_v6.h: event descriptor (ABI -- do not edit) ----------------
typedef struct {
  uint64_t type;
  void* parentObj;
  int rank;
  union {
    struct {
      bool graphCaptured;
      int groupDepth;
    } groupApi;

    struct {
      const char* func;
      size_t count;
      const char* datatype;
      int root;
      void* stream;
      bool graphCaptured;
    } collApi;

    struct {
      const char* func;
      size_t count;
      const char* datatype;
      void* stream;
      bool graphCaptured;
    } p2pApi;

    struct {
      void* stream;
    } kernelLaunch;

    struct {
      uint64_t seqNumber;
      const char* func;
      void const* sendBuff;
      void* recvBuff;
      size_t count;
      int root;
      const char* datatype;
      uint8_t nChannels;
      uint8_t nWarps;
      const char* algo;
      const char* proto;
      void* parentGroup;
    } coll;

    struct {
      const char* func;
      void* buff;
      const char* datatype;
      size_t count;
      int peer;
      uint8_t nChannels;
      void* parentGroup;
    } p2p;

    struct {
      pid_t pid;
      uint8_t channelId;
      int peer;
      int nSteps;
      int chunkSize;
      int isSend;
    } proxyOp;

    struct {
      int step;
    } proxyStep;

    struct {
      uint8_t channelId;
      uint64_t pTimer;
    } kernelCh;

    struct {
      int64_t id;
      void* data;
    } netPlugin;

    struct {
      uint64_t seqNumber;
      const char* func;
      void const* sendBuff;
      void* recvBuff;
      size_t count;
      int root;
      const char* datatype;
      const char* syncStrategy;
      bool intraBatchSync;
      uint32_t batchSize;
      uint32_t numBatches;
      uint32_t ceSeqNum;
      void* stream;
    } ceColl;

    struct {
      bool isComplete;
      int nRanks;
    } ceCollSync;

    struct {
      int numOps;
      size_t totalBytes;
      bool useIntraSync;
    } ceCollBatch;
  };
} ncclProfilerEventDescr_v6_t;

// --- profiler_v6.h: plugin vtable (ABI -- do not edit) -------------------
typedef struct {
  const char* name;

  ncclResult_t (*init)(
      void** context,
      uint64_t commId,
      int* eActivationMask,
      const char* commName,
      int nNodes,
      int nranks,
      int rank,
      ncclDebugLogger_t logfn);

  ncclResult_t (*startEvent)(
      void* context,
      void** eHandle,
      ncclProfilerEventDescr_v6_t* eDescr);

  ncclResult_t (*stopEvent)(void* eHandle);

  ncclResult_t (*recordEventState)(
      void* eHandle,
      ncclProfilerEventState_v6_t eState,
      ncclProfilerEventStateArgs_v6_t* eStateArgs);

  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v6_t;

} // extern "C"
