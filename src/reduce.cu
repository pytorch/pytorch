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

#include <algorithm>

#include <nvToolsExt.h>

#include "core.h"
#include "common_kernel.h"
#include "copy_kernel.h"
#include "enqueue.h"
#include "reduce_kernel.h"

/* HIERARCHY
 *
 * The data is split into CHUNKS, and each CHUNK is split into NUM_SUBCHUNKS
 * SUBCHUNKS, where each SUBCHUNK is processed independently. A SUBCHUNK is
 * split into numUnroll UNROLLS and each thread performs UNROLL_COUNT
 * single-data-element operations inside an UNROLL. As the name suggests, the
 * UNROLL_COUNT operations within an UNROLL are unrolled.
*/

// Number of threads used to perform copies, etc. Must be multiple of 32.
// An additional thread is used to handle threadfences, so the CUDA blocks
// have dimension NUM_THREADS+1.
#define NUM_THREADS     256

// Each thread unrolls the innermost loop of the copy or reduction operations
// to this many single-data-element instructions
#define UNROLL_COUNT    8

#define UNROLL_SIZE     (UNROLL_COUNT * NUM_THREADS)

// To hide the latency associated with the synchronization between different
// subchunks, we interleave the independent subchunks so that more data can be
// transferred while the sync is in progress. This is the number of subchunks
// that are active at the same time
#define NUM_SUBCHUNKS   4

// if this is called with CHUNK, it means that we just finished pushing the data
// of chunk CHUNK to the next GPU, so it can proceed with CHUNK
// We add 1 to chunk so that the initial flag of 0 doesn't allow the non-root
// GPUs to proceed before the flag is incremented from the upstream GPU. This
// is called by one particular consumer warp and so we select the first thread
// in the warp to set the flag.
#define SIGNAL_NEW_DATA_AVAILABLE(chunk, subchunk)                              \
    do {                                                                        \
      __threadfence_system();                                                   \
      args.NextNewDataAvailableFlag[0] = NUM_SUBCHUNKS*(chunk) + subchunk + 1;  \
    } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
#define WAIT_FOR_NEW_DATA(chunk, subchunk)                                      \
    do {                                                                        \
      if (tid == 0) {                                                           \
        Wait([=] {                                                              \
          return ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=          \
              NUM_SUBCHUNKS*(chunk) + subchunk + 1;                             \
        });                                                                     \
      }                                                                         \
      BAR(sync, 1, NUM_THREADS);                                                \
    } while (0)

// If this is called with CHUNK, it means that this GPU has just finished
// processing the chunk CHUNK and so the previous GPU can start with CHUNK + 1
#define SIGNAL_CHUNK_DONE(chunk, subchunk)                                      \
    do {                                                                        \
      args.PrevChunkDoneFlag[0] = NUM_SUBCHUNKS*(chunk) + subchunk + 1;         \
    } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
// all threads synchronize after thread 0 is done spinning.
#define WAIT_FOR_CHUNK(chunk, subchunk)                                         \
    do {                                                                        \
      if (tid == 0) {                                                           \
        Wait([=] {                                                              \
          return ((volatile int *)args.ThisChunkDoneFlag)[0] >=                 \
              NUM_SUBCHUNKS*(chunk) + subchunk + 1 - NUM_SUBCHUNKS;             \
        });                                                                     \
      }                                                                         \
      BAR(sync, 1, NUM_THREADS);                                                \
    } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
// all threads synchronize after thread 0 is done spinning.
#define WAIT_FOR_NEW_DATA_AND_CHUNK(chunk, subchunk)                            \
    do {                                                                        \
      if (tid == 0) {                                                           \
        Wait([=] {                                                              \
          bool newDataAvailable =                                               \
              ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=             \
                  NUM_SUBCHUNKS*(chunk) + subchunk + 1;                         \
          bool chunkDone =                                                      \
              ((volatile int *)args.ThisChunkDoneFlag)[0] >=                    \
                  NUM_SUBCHUNKS*(chunk)+subchunk + 1 - NUM_SUBCHUNKS;           \
          return newDataAvailable && chunkDone;                                 \
        });                                                                     \
      }                                                                         \
      BAR(sync, 1, NUM_THREADS);                                                \
    } while (0)

__device__ inline void getSliceSizeAndOffset(int *size, int *offset, int slice,
    int numSlices, int numBigSlices, int numSmallSlices, int bigSliceN,
    int smallSliceN, int lastSliceN) {
  if (slice < numBigSlices) {
    *size = bigSliceN;
    *offset = slice * bigSliceN;
  } else {
    *size = (slice < numBigSlices + numSmallSlices) ? smallSliceN
        : ((slice == numSlices - 1) ? lastSliceN : 0);
    *offset = numBigSlices * bigSliceN + (slice - numBigSlices) * smallSliceN;
  }

//  if (threadIdx.x == 0)
//    printf("[size=%d] [offset=%d] slice=%d numSlices=%d "
//        "numBigSlices=%d numSmallSlices=%d bigSliceN=%d smallSliceN=%d "
//        "lastSliceN=%d\n", *size, *offset, slice, numSlices, numBigSlices,
//        numSmallSlices, bigSliceN, smallSliceN, lastSliceN);
}

template<typename T>
struct ReduceKernelArgs {
  // general parameters
  int ThisId;
  int N;

  // some pre-computed sizes
  int SliceSize;
  int ChunkSize;
  int NumChunks;
  int BufferSliceStride;

  T ** ThisPtrToNextData;
  T ** PrevPtrToThisData;

  // local and remote data
  T * __restrict__ Output;
  const T * __restrict__ ThisData;
  volatile T * __restrict__ ThisBuffer;
  volatile T * __restrict__ NextBuffer;

  // local and remote flags
  volatile int * __restrict__ ThisNewDataAvailableFlag;
  volatile int * __restrict__ NextNewDataAvailableFlag;
  volatile int * __restrict__ ThisChunkDoneFlag;
  volatile int * __restrict__ PrevChunkDoneFlag;
};

__shared__ volatile void * nextData;
enum ReduceRole {BEGIN=0, MIDDLE=1, END=2};

template<int THREADS, int UNROLL, class FUNC, int ROLE, typename T>
__global__ void ReduceKernel(const ReduceKernelArgs<T> args) {
  if (args.N == 0) return;
  int tid = threadIdx.x;

  // First wait for args.PrevPtrToThisOutput to become nullptr to ensure that
  // the previous GPU is done with a previous collective operation.
  if (tid == 0) {
    if (ROLE != BEGIN) {
      Wait([=] {
        return *((T * volatile *)args.PrevPtrToThisData) == nullptr; // Wait for previous processor to be done
      });

      *((T * volatile *)args.PrevPtrToThisData) = (T*)args.ThisData; // Tell Previous I'm starting
    }
    if (ROLE != END) {
      Wait([=] {
        return *((T * volatile *)args.ThisPtrToNextData) != nullptr;  // Wait till I've been told next started
      });
    }
  }
  __syncthreads();

  for (int chunk = 0; chunk < args.NumChunks; ++chunk) {
    // calculate slice size.  for all chunks except (possibly) the last one,
    // this will just be args.SliceSize. For the last one, it may be smaller
    int bigSliceN   = args.SliceSize;
    int smallSliceN = 0;
    int lastSliceN  = 0;
    int numSlices   = NUM_SUBCHUNKS;
    int numBigSlices   = numSlices;
    int numSmallSlices = 0;

    // last chunk
    if ((chunk + 1 == args.NumChunks) && (args.N % args.ChunkSize > 0))
      CalcLastChunk<THREADS, UNROLL, T>(&bigSliceN, &smallSliceN, &lastSliceN,
          &numSlices, &numBigSlices, &numSmallSlices, args.N, args.NumChunks,
          args.ChunkSize);

    // this offset is only applied to Data pointers, not to Buffer pointers,
    // since we only have one buffer per chunk
    int chunkOffset = chunk * args.ChunkSize;

    int offset;
    int sliceSize;

    if (tid < THREADS) {
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        getSliceSizeAndOffset(&sliceSize, &offset, s, numSlices,
            numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

        if (ROLE == BEGIN) {
          WAIT_FOR_CHUNK(chunk, s);

          Copy<UNROLL, THREADS>(
              args.NextBuffer + (s * args.BufferSliceStride),
              args.ThisData + chunkOffset + offset,
              sliceSize);
        } else if (ROLE == MIDDLE) {
          WAIT_FOR_NEW_DATA_AND_CHUNK(chunk, s);

          Reduce<UNROLL, THREADS, FUNC>(
              args.NextBuffer + (s * args.BufferSliceStride),
              args.ThisData + chunkOffset + offset,
              args.ThisBuffer + (s * args.BufferSliceStride),
              sliceSize);
        } else { // ROLE == END
          WAIT_FOR_NEW_DATA(chunk, s);

          Reduce<UNROLL, THREADS, FUNC>(
              args.Output + chunkOffset + offset,
              args.ThisData + chunkOffset + offset,
              args.ThisBuffer + (s * args.BufferSliceStride),
              sliceSize);
        }
        __syncthreads();
      }
    } else { // Consumer thread
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        __syncthreads();
        if (ROLE != END)
          SIGNAL_NEW_DATA_AVAILABLE(chunk, s);

        // signal chunk done if we don't push into the receive buffer and this
        // is no the last chunk and this is not root
        if ((ROLE != BEGIN) && (chunk + 1 < args.NumChunks)) {
          SIGNAL_CHUNK_DONE(chunk, s);
        }
      }
    }
  }

  // reset flags
  if (tid == 0) {
    args.ThisNewDataAvailableFlag[0] = 0;
    args.ThisChunkDoneFlag[0] = 0;
    *args.ThisPtrToNextData = nullptr;
  }
}

template<class FUNC, typename T>
ncclResult_t ncclReduceWithTypeAndFunc(const void* sendbuff, void* recvbuff, 
    const int count, const int root, ncclComm* comm, cudaStream_t stream) {
  if (count == 0)
    return ncclSuccess;

  int index = comm->ncclId;

  const int numUnroll = 4;
  int rootId = comm->ringFromUser[root];

  int nextId = (index + 1) % comm->nDev;
  int prevId = (index + comm->nDev - 1) % comm->nDev;

  // There is one slice per GPU, so a slice can be at most bufferN / numGPUs,
  // where bufferN is the number of elements of type T that fit into the buffer.
  // For efficiency, we want the slice size to be a multiple of UNROLL_SIZE
  int bufferN = comm->buffSize / sizeof(T);
  // we only need buffer for k slices and k paddings
  int bufferNPerSlice = bufferN / NUM_SUBCHUNKS;
  int maxSliceSize = (bufferNPerSlice / UNROLL_SIZE) * UNROLL_SIZE;

  ReduceKernelArgs<T> args;

  args.ThisId = index;
  args.N = count;

  args.SliceSize = numUnroll * UNROLL_SIZE * sizeof(PackType) / sizeof(T);

  if(!comm->useRemoteRecv) {
    // Proxy for QPI. Reduce never pushes directly to recv.
    // But larger transfers help QPI more than tag updates hurt P2P.
    args.SliceSize *= 8;
  }

  // make sure slice fits into the temporary buffer
  args.SliceSize = std::min(maxSliceSize, args.SliceSize);
  args.BufferSliceStride = args.SliceSize;
  args.ChunkSize = NUM_SUBCHUNKS * args.SliceSize;

  // avoid a case where we have one or more big chunks and one tiny one
  int remainder = args.N % args.ChunkSize;
  if ((args.N > args.ChunkSize) && (remainder > 0) &&
      (args.N < 5 * args.ChunkSize) && (2 * remainder < args.ChunkSize)) {
    args.SliceSize /= 2;
    args.ChunkSize = NUM_SUBCHUNKS * args.SliceSize;

    // round down so we end up with a big last chunk
    args.NumChunks = args.N / args.ChunkSize;
  } else {
    // round up
    args.NumChunks = (args.N + args.ChunkSize - 1) / args.ChunkSize;
  }

  args.ThisPtrToNextData = (T**)&(comm->local[nextId]->recvPtrs[0]);
  args.PrevPtrToThisData = (T**)&(comm->remote[prevId]->recvPtrs[0]);

  args.Output = (T*)recvbuff;
  args.ThisData = (const T*) sendbuff;
  args.ThisBuffer = (volatile T*)comm->local[prevId]->buff;
  args.NextBuffer = (volatile T*)comm->remote[nextId]->buff;

  args.ThisNewDataAvailableFlag = comm->local[prevId]->flags;
  args.NextNewDataAvailableFlag = comm->remote[nextId]->flags;

  args.ThisChunkDoneFlag = comm->local[nextId]->flags + 1; 
  args.PrevChunkDoneFlag = comm->remote[prevId]->flags + 1;

  if (index == (rootId + 1) % comm->nDev) {
    ReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNC, BEGIN, T>
        <<<1, NUM_THREADS + 1, 0, stream>>>(args);
  } else if (index == rootId) {
    ReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNC, END, T>
        <<<1, NUM_THREADS + 1, 0, stream>>>(args);
  } else {
    ReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNC, MIDDLE, T>
        <<<1, NUM_THREADS + 1, 0, stream>>>(args);
  }
  return ncclSuccess;
}

template <typename T>
ncclResult_t ncclReduceWithType(const void* sendbuff,
      void* recvbuff, int count, ncclRedOp_t op, int root,
      ncclComm* comm, cudaStream_t stream) {

  switch (op) {
    case ncclSum:
      return ncclReduceWithTypeAndFunc<FuncSum<T>, T>(
          sendbuff, recvbuff, count, root, comm, stream);
    case ncclProd:
      return ncclReduceWithTypeAndFunc<FuncProd<T>, T>(
          sendbuff, recvbuff, count, root, comm, stream);
    case ncclMax:
      return ncclReduceWithTypeAndFunc<FuncMax<T>, T>(
          sendbuff, recvbuff, count, root, comm, stream);
    case ncclMin:
      return ncclReduceWithTypeAndFunc<FuncMin<T>, T>(
          sendbuff, recvbuff, count, root, comm, stream);
  }
  return ncclInvalidOperation;
}


class ReduceFunctor {
public:
  ncclResult_t operator()(const void* sendbuff,
      void* recvbuff, int count, ncclDataType_t datatype, ncclRedOp_t op,
      int root, ncclComm* comm, cudaStream_t stream) {

    switch (datatype) {
    case ncclChar:
      return ncclReduceWithType<char>(sendbuff, recvbuff, count, op, root, comm, stream);
    case ncclInt:
      return ncclReduceWithType<int>(sendbuff, recvbuff, count, op, root, comm, stream);
#ifdef CUDA_HAS_HALF
    case ncclHalf:
      return ncclReduceWithType<half>(sendbuff, recvbuff, count, op, root, comm, stream);
#endif
    case ncclFloat:
      return ncclReduceWithType<float>(sendbuff, recvbuff, count, op, root, comm, stream);
    case ncclDouble:
      return ncclReduceWithType<double>(sendbuff, recvbuff, count, op, root, comm, stream);
    }
    return ncclInvalidType;
  }
};

extern "C" DSOGLOBAL
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm,
    cudaStream_t stream) {
  return enqueue(ReduceFunctor(), sendbuff, recvbuff, count, datatype, op,
      root, comm, stream);
}

