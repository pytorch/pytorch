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

#include <nvToolsExt.h>

#include "core.h"
#include "common_kernel.h"
#include "copy_kernel.h"
#include "enqueue.h"
#include "reduce_kernel.h"

/* HIERARCHY
 *
 * The data is split into CHUNKS, and each CHUNK is split into NUM_SUBCHUNKS
 * SUBCHUNKS, where each SUBCHUNK is an independent, complete reduction. Each
 * GPU has a buffer that can fit an entire CHUNK, so that all SUBCHUNKS can be
 * processed without checking that the buffer on the receiving GPU is empty. A
 * SUBCHUNK is split into NUM_GPUS SLICES and each GPU works on a different
 * SLICE at the same time. Before moving on the the next SLICE in the reduction
 * algorithm, the GPU has to check whether it has received the data from the
 * previous GPU it needs for this SLICE. To hide the latency of this
 * communication, each GPU processes all the SLICES of all the SUBCHUNKS in
 * sequence before moving on to the next SLICE. Each SLICE is split into a
 * certain number of UNROLLS (determined by the buffer size) and each thread
 * performs UNROLL_COUNT single-data-element operations inside an UNROLL. As the
 * name suggests, the UNROLL_COUNT operations within an UNROLL are unrolled.
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
#define NUM_SUBCHUNKS   2


// If this is called with STEP, it means that we just finished processing the
// data for step STEP on this GPU, which is the data required on the next GPU
// for step STEP + 1, so we signal the next GPU that its data for step STEP + 1
// is available. This is called by one particular consumer warp and so we select
// the first thread in the warp to set the flag.
#define SIGNAL_NEW_DATA_AVAILABLE(chunk, subchunk, step)                      \
    do {                                                                      \
      __threadfence_system();                                                 \
      args.NextNewDataAvailableFlag[0] =                                      \
          NUM_SUBCHUNKS*((chunk) * (2 * args.NumGPUs - 2) + (step) + 1)+subchunk;         \
    } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
#define WAIT_FOR_NEW_DATA(chunk, subchunk, step)                                \
    do {                                                                        \
      if (tid == 0) {                                                           \
        Wait([=] {                                                              \
          return ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=          \
              2*((chunk) * (2 * args.NumGPUs - 2) + (step))+subchunk;           \
        });                                                                     \
      }                                                                         \
      BAR(sync, 1, NUM_THREADS);                                                \
    } while (0)

#define SIGNAL_CHUNK_DONE(chunk, subchunk)                                      \
    do {                                                                        \
      args.PrevChunkDoneFlag[0] = NUM_SUBCHUNKS*(chunk) + subchunk + 1;         \
    } while (0)

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
}

template<typename T>
struct AllReduceKernelArgs {
  // general parameters
  int ThisId;
  int NumGPUs;
  int N;

  // some pre-computed sizes
  int SliceSize;
  int ChunkSize;
  int NumChunks;

  T ** ThisPtrToNextOutput;
  T ** PrevPtrToThisOutput;

  // local and remote input, output, and buffer
  const T * __restrict__ ThisInput;
  volatile T * __restrict__ ThisOutput;
  volatile T * __restrict__ ThisBuffer;
  volatile T * __restrict__ NextBuffer;

  // local and remote flags
  volatile int * __restrict__ ThisNewDataAvailableFlag;
  volatile int * __restrict__ NextNewDataAvailableFlag;
  volatile int * __restrict__ ThisChunkDoneFlag;
  volatile int * __restrict__ PrevChunkDoneFlag;
};

__shared__ volatile void * nextOutput;


template<int THREADS, int UNROLL, class FUNC, bool PUSHRECV, typename T>
__launch_bounds__(THREADS+WARP_SIZE, 1)
__global__ void AllReduceKernel(const AllReduceKernelArgs<T> args) {
  if (args.N == 0) return;
  const int tid = threadIdx.x;

  // First wait for args.PrevPtrToThisOutput to become nullptr to ensure that
  // the previous GPU is done with a previous collective operation.
  if (tid == 0) {
    Wait([=] {
      return *((T * volatile *)args.PrevPtrToThisOutput) == nullptr;
    });

    *((T * volatile *)args.PrevPtrToThisOutput) = (T*)args.ThisOutput;

    Wait([=] {
      return *((T * volatile *)args.ThisPtrToNextOutput) != nullptr;
    });

    if (PUSHRECV)
      nextOutput =
        *((volatile void * volatile *)args.ThisPtrToNextOutput);
  }
  __syncthreads();


  for (int chunk = 0; chunk < args.NumChunks; ++chunk) {
    // calculate slice size.  for all chunks except (possibly) the last one,
    // this will just be args.SliceSize. For the last one, it may be smaller
    int bigSliceN   = args.SliceSize;
    int smallSliceN = 0;
    int lastSliceN  = 0;
    int numSlices   = args.NumGPUs * NUM_SUBCHUNKS;
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

    /////////////// begin AllReduce steps ///////////////

    // step 0: push data to next GPU
    int step = 0;
    int slice = args.ThisId;
    int offset;
    int sliceSize;

    if (tid < THREADS) {
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        if (s > 0) { slice += args.NumGPUs; }
        getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
            numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

        if (!PUSHRECV && chunk > 0) {
          WAIT_FOR_CHUNK(chunk, s);
        }

        Copy<UNROLL, THREADS>(
            args.NextBuffer + offset,
            args.ThisInput + chunkOffset + offset,
            sliceSize);

        __syncthreads();
      }
    } else { // is consumer thread
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
      }
    }

    // steps j with 1 <= j < k - 1, where k = number of GPUs:
    // reduce and copy to next GPU
    for (step = 1; step < args.NumGPUs - 1; ++step) {
      if (tid < THREADS) {
        slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          if (s > 0) { slice += args.NumGPUs; }
          getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
              numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

          WAIT_FOR_NEW_DATA(chunk, s, step);

          Reduce<UNROLL, THREADS, FUNC>(
              args.NextBuffer + offset,
              args.ThisBuffer + offset,
              args.ThisInput + chunkOffset + offset,
              sliceSize);

          __syncthreads();
        }
      } else {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
        }
      }
    }

    // step k - 1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    step = args.NumGPUs - 1;

    if (tid < THREADS) {
      slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        if (s > 0) { slice += args.NumGPUs; }
        getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
            numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

        WAIT_FOR_NEW_DATA(chunk, s, step);

        if (PUSHRECV) {
          ReduceAndCopy<UNROLL, THREADS, FUNC>(
              (volatile T *)nextOutput + chunkOffset + offset,
              args.ThisOutput + chunkOffset + offset,
              args.ThisBuffer + offset,
              args.ThisInput + chunkOffset + offset,
              sliceSize);
        } else {
          ReduceAndCopy<UNROLL, THREADS, FUNC>(
              args.NextBuffer + offset,
              args.ThisOutput + chunkOffset + offset,
              args.ThisBuffer + offset,
              args.ThisInput + chunkOffset + offset,
              sliceSize);
        }

        __syncthreads();
      }
    } else {
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
      }
    }

    // steps j with k <= j < 2*k-2: copy result to next GPU
    for (step = args.NumGPUs; step < 2 * args.NumGPUs - 2; ++step) {
      if (tid < THREADS) {
        slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          if (s > 0) { slice += args.NumGPUs; }
          getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
              numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

          WAIT_FOR_NEW_DATA(chunk, s, step);

          if( PUSHRECV ) {
            Copy<UNROLL, THREADS>(
                (volatile T *)nextOutput + chunkOffset + offset,
                args.ThisOutput + chunkOffset + offset,
                sliceSize);
          } else {
            DoubleCopy<UNROLL, THREADS>(
                args.NextBuffer + offset,
                args.ThisOutput + chunkOffset + offset,
                args.ThisBuffer + offset,
                sliceSize);
          }

          __syncthreads();
        }
      } else {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
        }
      }
    }

    if (!PUSHRECV) {
      // Make final copy from buffer to dest.
      if (tid < THREADS) {
        slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          if (s > 0) { slice += args.NumGPUs; }
          getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
              numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

          WAIT_FOR_NEW_DATA(chunk, s, step);

          // Here we need to copy from buffer to this output.
          Copy<UNROLL, THREADS>(
              args.ThisOutput + chunkOffset + offset,
              args.ThisBuffer + offset,
              sliceSize);

          __syncthreads();
        }
      } else {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          if(chunk+1 < args.NumChunks) {
            SIGNAL_CHUNK_DONE(chunk, s);
          }
        }
      }
    }
  }

  // wait for the last data to be pushed to us
  if (tid < THREADS) {
    if(PUSHRECV) {
      WAIT_FOR_NEW_DATA(args.NumChunks, NUM_SUBCHUNKS-1, 0);
    }

    if (tid == 0) {
      args.ThisNewDataAvailableFlag[0] = 0;
      if(!PUSHRECV) {
        args.ThisChunkDoneFlag[0] = 0;
      }
      *args.ThisPtrToNextOutput = nullptr;
    }
  }
}

template<class FUNC, typename T>
ncclResult_t ncclAllReduceWithTypeAndFunc(const void* sendbuff, void* recvbuff,
    const int count, ncclComm* comm, cudaStream_t stream) {
  if (count == 0)
    return ncclSuccess;
  int index = comm->ncclId;

  // There is one slice per GPU, so a slice can be at most bufferN / numGPUs,
  // where bufferN is the number of elements of type T that fit into the buffer.
  // For efficiency, we want the slice size to be a multiple of UNROLL_SIZE
  int bufferN = comm->buffSize / sizeof(T);
  int bufferNPerSlice = bufferN / (NUM_SUBCHUNKS * comm->nDev);
  int sliceSize = (bufferNPerSlice / UNROLL_SIZE) * UNROLL_SIZE;

  int nextId = (index + 1) % comm->nDev;
  int prevId = (index + comm->nDev - 1) % comm->nDev;

  AllReduceKernelArgs<T> args;

  args.ThisId = index;
  args.NumGPUs = comm->nDev;
  args.N = count;

  args.SliceSize = sliceSize;
  int subchunkSize = comm->nDev * args.SliceSize;
  args.ChunkSize = NUM_SUBCHUNKS * subchunkSize;

  // avoid a case where we have one or more big chunks and one tiny one
  int remainder = args.N % args.ChunkSize;
  if ((args.N > args.ChunkSize) && (remainder > 0) &&
      (args.N < 5 * args.ChunkSize) && (2 * remainder < args.ChunkSize)) {
    args.SliceSize /= 2;
    int subchunkSize = comm->nDev * args.SliceSize;
    args.ChunkSize = NUM_SUBCHUNKS * subchunkSize;

    // round down so we end up with a big last chunk
    args.NumChunks = args.N / args.ChunkSize;
  } else {
    // round up
    args.NumChunks = (args.N + args.ChunkSize - 1) / args.ChunkSize;
  }

  args.ThisPtrToNextOutput = (T**)&(comm->local[nextId]->recvPtrs[0]);
  args.PrevPtrToThisOutput = (T**)&(comm->remote[prevId]->recvPtrs[0]);

  args.ThisInput = (const T*)sendbuff;
  args.ThisOutput = (volatile T*)recvbuff;
  args.ThisBuffer = (volatile T*)comm->local[prevId]->buff;
  args.NextBuffer = (volatile T*)comm->remote[nextId]->buff;

  args.ThisNewDataAvailableFlag = comm->local[prevId]->flags;
  args.NextNewDataAvailableFlag = comm->remote[nextId]->flags;

  args.ThisChunkDoneFlag = comm->local[nextId]->flags + 1; 
  args.PrevChunkDoneFlag = comm->remote[prevId]->flags + 1;

  if( comm->useRemoteRecv ) {
    AllReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNC, true, T>
        <<<1, NUM_THREADS + 1, 0, stream>>>(args);
  } else {
    AllReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNC, false, T>
        <<<1, NUM_THREADS + 1, 0, stream>>>(args);
  }
  return ncclSuccess;
}


template<typename T>
ncclResult_t ncclAllReduceWithType(const void* sendbuff,
    void* recvbuff, int count, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  switch (op) {
  case ncclSum:
    return ncclAllReduceWithTypeAndFunc<FuncSum<T>, T>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclProd:
    return ncclAllReduceWithTypeAndFunc<FuncProd<T>, T>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclMax:
    return ncclAllReduceWithTypeAndFunc<FuncMax<T>, T>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclMin:
    return ncclAllReduceWithTypeAndFunc<FuncMin<T>, T>(
        sendbuff, recvbuff, count, comm, stream);
  }
  return ncclInvalidOperation;
}

class AllReduceFunctor {
public:
  ncclResult_t operator()(const void* sendbuff, void* recvbuff,
      int count, ncclDataType_t datatype, ncclRedOp_t op, int /*root*/,
      ncclComm* comm, cudaStream_t stream) {

    switch (datatype) {
    case ncclChar:
      return ncclAllReduceWithType<char>(sendbuff, recvbuff, count, op,
          comm, stream);
    case ncclInt:
      return ncclAllReduceWithType<int>(sendbuff, recvbuff, count, op,
          comm, stream);
#if CUDART_VERSION >= 7050
    case ncclHalf:
      return ncclAllReduceWithType<half>(sendbuff, recvbuff, count, op,
          comm, stream);
#endif
    case ncclFloat:
      return ncclAllReduceWithType<float>(sendbuff, recvbuff, count, op,
          comm, stream);
    case ncclDouble:
      return ncclAllReduceWithType<double>(sendbuff, recvbuff, count, op,
          comm, stream);
    }

    return ncclInvalidType;
  }
};

extern "C" DSOGLOBAL
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  return enqueue(AllReduceFunctor(), sendbuff, recvbuff, count, datatype, op, 0,
      comm, stream);
}

