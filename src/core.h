/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#ifndef CORE_H_
#define CORE_H_

#include "nccl.h"
#include <cstdio>
#include <cuda_runtime.h>

#define MAXFLAGS 8
#define MAXQUEUE 4 // Maximum number of queued collectives per communicator.
#define DEFAULT_BUFFER_SIZE_BYTES (1UL << 21)

// DIE on error
#define CUDACHECK(cmd) do {                              \
    cudaError_t e = cmd;                                 \
    if( e != cudaSuccess ) {                             \
        printf("Cuda failure %s:%d '%s'\n",              \
               __FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                              \
    }                                                    \
} while(false)

#define NCCL_MEM_PAD_ALIGN 4096

typedef struct {
  cudaEvent_t isDone[MAXQUEUE];
  int back; // Last event used
} EventQueue;

struct ncclMem {
  union { // Pad this block so that devBuff is correctly aligned
    struct {
      int   flags[MAXFLAGS];
      void* recvPtrs[MAXFLAGS];
    };
    char pad[NCCL_MEM_PAD_ALIGN];
  };
  // devBuff will likely be bigger ; we only use its offset/address.
  char buff[NCCL_MEM_PAD_ALIGN];
};

struct ncclNodeRef {
  ncclMem* remote;
  ncclMem* local;
  int remoteCleanup;
  void* cleanupHandle;
};

struct ncclComm {
  int nDev;    // number of devices in communicator
  int cudaDev; // cuda device index
  int ncclId;  // nccl logical index

  // Device and Host allocated chunks. Stored here to correctly free() memory.
  ncclMem* devMem;
  ncclMem* hostMem;
  int hostMemState;

  // Placed between calling and internal device streams.
  EventQueue events;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices.
  int* userFromRing;

  // copy of the above stored on each device
  int* devUserFromRing;

  // Inverse of userFromRing. Maps user specified index to internal nccl index.
  int* ringFromUser;

  // Size of temp buffer in bytes.
  size_t buffSize;

  // Whether we have remote access to the recvbuff pointers passed from remote
  // GPUs. In single process mode this can be used as long as QPI links are
  // not present. In multi-process, we never push to a remote recvbuff.
  int useRemoteRecv;

  // Device-to-device communication structures to access remote or local device
  // memory. Actual allocation larger than 1.
  ncclNodeRef ptrs[1];
};

typedef enum {NONE=0, VERSION=1, WARN=2, INFO=3, ABORT=4} DebugLevel;
extern DebugLevel ncclDebugLevel;

#define WARN(...) do {                                           \
  if (ncclDebugLevel >= WARN) {                                  \
    printf("WARN %s:%d ", __FILE__, __LINE__);                   \
    printf(__VA_ARGS__);                                         \
    printf("\n");                                                \
    if (ncclDebugLevel >= ABORT) abort();                        \
  }                                                              \
} while(0)

#define INFO(...) do {                                           \
  if (ncclDebugLevel >= INFO) {                                  \
    printf("INFO "); printf(__VA_ARGS__); printf("\n");          \
  }                                                              \
} while(0)

#define DSOGLOBAL __attribute__((visibility("default")))

#endif // end include guard

