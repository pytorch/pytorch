/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef CORE_H_
#define CORE_H_


#include "nccl.h"
#include <cstdio>
#include <cuda_runtime.h>

#define MAXRANKS 32
#define DEFAULT_BUFFER_SIZE_BYTES (1UL << 21)
#define NCCL_MEM_PAD_ALIGN 65536


struct ncclMem {
  union { // Pad this block so that devBuff is correctly aligned
    struct {
      int   flags[2];
      void* recvPtrs;
      int   opCounter; // Used to determine when remote Communicators are ready.
                       // Only used in host memory.
    };
    char pad[NCCL_MEM_PAD_ALIGN];
  };
  // devBuff will be bigger ; we only use its offset/address.
  char buff[1];
};

template <typename T>
struct alignas(long long) DevRing {
  volatile int* __restrict__ prevOpCounter;
  volatile int* __restrict__ nextOpCounter;
  volatile int* __restrict__ sendFlagToNext;
  volatile int* __restrict__ sendFlagToPrev;
  volatile int* __restrict__ recvFlagFromNext;
  volatile int* __restrict__ recvFlagFromPrev;

  T* volatile * __restrict__ recvPtrFromNext;
  T* volatile * __restrict__ sendPtrToPrev;
  T*   __restrict__ recvBuffer;
  T*   __restrict__ sendBuffer;

  int userRank[MAXRANKS];
};

struct NodeRef {
  ncclMem* remote; // TODO: Verify if these
  ncclMem* local;  //       are still needed.
  enum {DEVICE, HOST} type;
  ncclMem* devCleanup;  // Used only when remote comm uses same process & GPU
  ncclMem* hostCleanup; // Used whenever target is in different process
  int* opCounter; // TODO: see if this can be removed too.
};


struct ncclComm {
  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index

  // Device and Host allocated chunks. Stored here to correctly free() memory.
  ncclMem* devMem;
  ncclMem* hostMem;
  int hostMemState;
  int opSched; // Scheduling operation index
  int* opCounter; // Counter of completed operations

  cudaStream_t prevStream; // cache last used stream
  cudaEvent_t doneEvent; // orders operations in different streams

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userFromRing;

  // copy of the above stored on each device
  int* devUserFromRing;

  // Ring order
  int* ncclFromRing; // TODO: REMOVE IF NOT NEEDED BEYOND CORE.CU

  // Size of temp buffer in bytes.
  size_t buffSize;

  // Whether we have remote access to the recvbuff pointers passed from remote
  // GPUs. In single process mode this can be used as long as QPI links are
  // not present. In multi-process, we never push to a remote recvbuff.
  int globalMemSpace;

  // Device copy of the communicator
  struct ncclComm *devComm;  // TODO: Remove this if not useful

  // Device-side ring view
  DevRing<char>* devRing;

  // Device-to-device communication structures to access remote or local device
  // memory. Actual allocation larger than 1.
  NodeRef ptrs[1];
};


typedef enum {NONE=0, VERSION=1, WARN=2, INFO=3, ABORT=4} DebugLevel;
extern DebugLevel ncclDebugLevel;

#define WARN(...) do {                                           \
  if (ncclDebugLevel >= WARN) {                                  \
    printf("WARN %s:%d ", __FILE__, __LINE__);                   \
    printf(__VA_ARGS__);                                         \
    printf("\n");                                                \
    fflush(stdout);                                              \
    if (ncclDebugLevel >= ABORT) abort();                        \
  }                                                              \
} while(0)

#define INFO(...) do {                                           \
  if (ncclDebugLevel >= INFO) {                                  \
    printf("INFO "); printf(__VA_ARGS__); printf("\n");          \
    fflush(stdout);                                              \
  }                                                              \
} while(0)

// Check CUDA calls
#define CUDACHECK(cmd, retcode) do {                        \
    cudaError_t e = cmd;                                    \
    if( e != cudaSuccess ) {                                \
        WARN("Cuda failure '%s'\n", cudaGetErrorString(e)); \
        return retcode;                                     \
    }                                                       \
} while(false)

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    return res; \
  } \
} while (0);

#ifdef PROFAPI
#define NCCL_API(ret, func, args...)        \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((alias(#func)))          \
    ret p##func (args);                     \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((weak))                  \
    ret func(args)
#else
#define NCCL_API(ret, func, args...)        \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    ret func(args)
#endif // end PROFAPI


#endif // end include guard

