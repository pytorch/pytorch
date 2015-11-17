/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

#ifndef CORE_H_
#define CORE_H_

#include "nccl.h"
#include <cstdio>
#include <cuda_runtime.h>

#define MAXPEERS 8 // Maximum number of devices.
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

struct ncclComm {
  int nDev;    // number of devices in communicator
  int cudaDev; // cuda device index
  int ncclId;  // nccl logical index

  // Device and Host allocated chunks. Stored here to correctly free() memory.
  ncclMem* devMem;
  ncclMem* hostMem;
  int hostMemState;

  // Device-to-device communication structures to access remote or local device
  // memory.
  ncclMem* remote[MAXPEERS];
  ncclMem* local[MAXPEERS];
  struct {
    int type;
    void* handle;
  } cleanup[MAXPEERS];
  //int remoteCleanup[MAXPEERS]; // 0 is raw ptr, 1 is unregister/unmap, 2 is ipc close

  // Placed between calling and internal device streams.
  EventQueue events;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices.
  int userFromRing[MAXPEERS];

  // copy of the above stored on each device
  int* devUserFromRing;

  // Inverse of userFromRing. Maps user specified index to internal nccl index.
  int ringFromUser[MAXPEERS];

  // Size of temp buffer in bytes.
  size_t buffSize;

  // Whether we have remote access to the recvbuff pointers passed from remote
  // GPUs. In single process mode this can be used as long as QPI links are
  // not present. In multi-process, we never push to a remote recvbuff.
  int useRemoteRecv;
};

typedef enum {NONE=0, WARN=1, INFO=2, ABORT=3} DebugLevel;
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

