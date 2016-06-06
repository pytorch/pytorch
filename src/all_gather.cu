/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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
#include <cassert>

#include "core.h"
#include "common_kernel.h"
#include "copy_kernel.h"
#include "enqueue.h"

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
#define NUM_SUBCHUNKS   2

// If this is called with STEP, it means that we just finished processing the
// data for step STEP on this GPU, which is the data required on the next GPU
// for step STEP + 1, so we signal the next GPU that its data for step STEP + 1
// is available. This is called by one particular consumer warp and so we select
// the first thread in the warp to set the flag.
#define SIGNAL_NEW_DATA_AVAILABLE(chunk, subchunk, step)                        \
    do {                                                                        \
      __threadfence_system();                                                   \
      args.NextNewDataAvailableFlag[0] =                                        \
            NUM_SUBCHUNKS*((chunk) * (args.NumGPUs - 1) + (step)) + subchunk+1; \
    } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
#define WAIT_FOR_NEW_DATA(chunk, subchunk, step)                               \
    do {                                                                       \
      if (tid == 0) {                                                          \
        Wait([=] {                                                             \
          return ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=         \
              NUM_SUBCHUNKS*((chunk) * (args.NumGPUs - 1) + (step))            \
              + subchunk + 1 - NUM_SUBCHUNKS;                                  \
        });                                                                    \
      }                                                                        \
      BAR(sync, 1, NUM_THREADS);                                               \
    } while (0)

#define SIGNAL_CHUNK_DONE(chunk, subchunk)                                     \
    do {                                                                       \
      __threadfence_system();                                                  \
      args.PrevChunkDoneFlag[0] = NUM_SUBCHUNKS*(chunk) + (subchunk) + 1;      \
    } while (0)

#define WAIT_FOR_PREV_CHUNK(chunk, subchunk)                       \
    do {                                                           \
      if (tid == 0) {                                              \
        Wait([=] {                                                 \
          return ((volatile int*)args.ThisChunkDoneFlag)[0] >=     \
              NUM_SUBCHUNKS*(chunk) + subchunk + 1-NUM_SUBCHUNKS;  \
        });                                                        \
      }                                                            \
      BAR(sync, 1, NUM_THREADS);                                   \
    } while (0)

__device__ inline void getSliceSizeAndChunkSize(int *sliceSize, int slice,
    int numSlices, int numBigSlices, int numSmallSlices, int bigSliceN,
    int smallSliceN, int lastSliceN) {
  if (slice < numBigSlices) {
    *sliceSize = bigSliceN;
  } else {
    *sliceSize = (slice < numBigSlices + numSmallSlices) ? smallSliceN
        : ((slice == numSlices - 1) ? lastSliceN : 0);
  }
}

template<typename T>
struct AllGatherKernelArgs {
  // general parameters
  int ThisId;
  int NumGPUs;
  int N;
  int * UserFromRing;

  // some pre-computed sizes
  int SliceSize;
  int ChunkSize;
  int NumChunks;

  int BufferSliceStride;
  int BufferMisalignedN;

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

__device__ inline int GetBlock(const int index, const int step,
    const int * const userFromRing, const int numGPUs) {
  return userFromRing[(numGPUs + index - step) % numGPUs];
}

__shared__ volatile void * nextOutput;

template<int THREADS, int UNROLL, bool PUSHRECV, typename T>
__global__ void AllGatherKernel(const AllGatherKernelArgs<T> args) {
  if (args.N == 0) return;
  int tid = threadIdx.x;

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

    if(PUSHRECV)
      nextOutput = *((volatile void * volatile *)args.ThisPtrToNextOutput);
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

    // step 0: copy the resident block from the ThisInput to ThisOutput and also
    // to NextOutput
    int step = 0;
    int block = GetBlock(args.ThisId, step, args.UserFromRing, args.NumGPUs);
    int outputOffset = chunkOffset + block * args.N;
    int inputOffset = chunkOffset;
    int bufferOffset;
    int sliceSize;

    if (!PUSHRECV) {
      bufferOffset = block * NUM_SUBCHUNKS * args.BufferSliceStride +
          block * args.BufferMisalignedN;
    }

    // Copy from ThisInput
    if (tid < THREADS) {
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        getSliceSizeAndChunkSize(&sliceSize, s, numSlices, numBigSlices,
            numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

        if (!PUSHRECV)
          WAIT_FOR_PREV_CHUNK(chunk, s);

        if (PUSHRECV) {
          DoubleCopy<UNROLL, THREADS>(
              args.ThisOutput + outputOffset,
              (volatile T *)nextOutput + outputOffset,
              args.ThisInput + inputOffset,
              sliceSize);
        } else {
          DoubleCopy<UNROLL, THREADS>(
              args.ThisOutput + outputOffset,
              args.NextBuffer + bufferOffset,
              args.ThisInput + inputOffset,
              sliceSize);
        }
        __syncthreads();

        outputOffset += sliceSize;
        inputOffset += sliceSize;
        if (!PUSHRECV)
          bufferOffset += sliceSize;
      }
    } else {
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
      }
    }

    // steps j with 0 < j < k - 1:
    // copy a block that was pushed to this GPU to the next GPU
    for (step = 1; step < args.NumGPUs - 1; ++step) {
      block = GetBlock(args.ThisId, step, args.UserFromRing, args.NumGPUs);
      outputOffset = chunkOffset + block * args.N;
      if (!PUSHRECV) {
        bufferOffset = block * NUM_SUBCHUNKS * args.BufferSliceStride +
            block * args.BufferMisalignedN;
      }

      if (tid < THREADS) {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          getSliceSizeAndChunkSize(&sliceSize, s, numSlices, numBigSlices,
              numSmallSlices, bigSliceN, smallSliceN, lastSliceN);
          WAIT_FOR_NEW_DATA(chunk, s, step);

          if (PUSHRECV) {
            Copy<UNROLL, THREADS>(
                (volatile T *)nextOutput + outputOffset,
                args.ThisOutput + outputOffset,
                sliceSize);
          } else {
            DoubleCopy<UNROLL, THREADS>(
                args.NextBuffer + bufferOffset,
                args.ThisOutput + outputOffset,
                args.ThisBuffer + bufferOffset,
                sliceSize);
          }
          __syncthreads();

          outputOffset += sliceSize;
          if (!PUSHRECV)
            bufferOffset += sliceSize;
        }
      } else {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
        }
      }
    }

    if (!PUSHRECV) {
      step = args.NumGPUs - 1;
      block = GetBlock(args.ThisId, step, args.UserFromRing, args.NumGPUs);
      outputOffset = chunkOffset + block * args.N;
      bufferOffset = block * NUM_SUBCHUNKS * args.BufferSliceStride +
          block * args.BufferMisalignedN;

      // Make final copy from buffer to dest.
      if (tid < THREADS) {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          getSliceSizeAndChunkSize(&sliceSize, s, numSlices, numBigSlices,
              numSmallSlices, bigSliceN, smallSliceN, lastSliceN);
          WAIT_FOR_NEW_DATA(chunk, s, step);

          Copy<UNROLL, THREADS>(
              args.ThisOutput + outputOffset,
              args.ThisBuffer + bufferOffset,
              sliceSize);

          __syncthreads();

          outputOffset += sliceSize;
          bufferOffset += sliceSize;
        }
      } else {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          SIGNAL_CHUNK_DONE(chunk, s);
        }
      }
    }
  }

  // wait for the last data to be pushed to us
  if (tid < THREADS) {
    if (PUSHRECV)
      WAIT_FOR_NEW_DATA(args.NumChunks, NUM_SUBCHUNKS-1, 0);
    else
      WAIT_FOR_PREV_CHUNK(args.NumChunks, NUM_SUBCHUNKS-1);

    if (tid == 0) {
      args.ThisNewDataAvailableFlag[0] = 0;
      args.ThisChunkDoneFlag[0] = 0;
      *args.ThisPtrToNextOutput = nullptr;
    }
  }
}

template<typename T>
ncclResult_t ncclAllGatherWithType(const void* sendbuff, void* recvbuff,
    int count, ncclComm* comm, int numUnroll, cudaStream_t stream) {
  if (count == 0)
      return ncclSuccess;

  int index = comm->ncclId;

  int blockSizeInBytes = count * sizeof(T);
  int misalignedBytes = blockSizeInBytes % alignof(uint64_t);

  assert((int)((misalignedBytes / sizeof(T)) * sizeof(T)) == misalignedBytes);

  int misalignedN = misalignedBytes / sizeof(T);
  assert(misalignedN < (int)(sizeof(uint64_t) / sizeof(T)));

  int paddingN = (misalignedN > 0) ? sizeof(uint64_t) / sizeof(T) : 0;

  // There is one slice per GPU, so a slice can be at most bufferN / numGPUs,
  // where bufferN is the number of elements of type T that fit into the buffer.
  int bufferN = comm->buffSize / sizeof(T);
  // we only need buffer for k slices and k paddings
  int bufferNPerSlice = (bufferN - comm->nDev * NUM_SUBCHUNKS * paddingN)
       / (comm->nDev * NUM_SUBCHUNKS);
  // For efficiency, we want the slice size to be a multiple of UNROLL_SIZE
  int maxSliceSize = (bufferNPerSlice / UNROLL_SIZE) * UNROLL_SIZE;
  int nextId = (index + 1) % comm->nDev;
  int prevId = (index + comm->nDev - 1) % comm->nDev;

  AllGatherKernelArgs<T> args;

  args.ThisId = index;
  args.NumGPUs = comm->nDev;
  args.N = count;

  /* Block j is coming from sendbuff[j], which lives on device with logical
   * index comm->ringFromUser[j]. But the block ordering does not necessarily
   * follow the ring ordering. Hence the order in which a particular GPU
   * processes the different blocks (the correspondence between the step in
   * the reduction algorithm and the block on which a GPU operates in that
   * particular step) is not the same as the ring order.
   *
   * Say we have 4 GPUs and comm->userFromRing = { 1, 2, 0, 3 }. Then there are 3
   * step in the all-gather algorithm and block 0 comes from device 2, block 1
   * from 0, block 2 from device 1, and block 3 comes from device 3. In the
   * first step of the algorithm, each GPU must copy its own block from its
   * sendbuff to the appropriate location in its recvbuff. The blocks that a
   * GPU has to process in the next steps is determined by the previous step
   * because each GPU only hands off data to the next GPU in the ring.
   *
   * In the above example, we get the following table of which block is
   * processed by each GPU in a given step. The columns correspond to the
   * different GPUs while the rows are the steps in the algorithm.
   *
   *      GPU 0   1   2   3
   * step
   *    0     1   2   0   3
   *    1     3   1   2   0
   *    2     0   3   1   2
   *
   * We note the the rows in the above table are just comm->userFromRing in the
   * first step and the list is cyclicly permuted to the right for each next
   * step. The columns, which are what the individual GPUs need to know, are
   * comm->userFromRing traversed backwards and starting at index k for GPU k.
   * These columns are what we put into args.BlockVsStep to tell the GPU which
   * block it needs to be processing at a particular step. */
  args.UserFromRing = comm->devUserFromRing;

  args.SliceSize = numUnroll * UNROLL_SIZE * sizeof(PackType) / sizeof(T);
  args.SliceSize = std::min(maxSliceSize, args.SliceSize);
  args.ChunkSize = NUM_SUBCHUNKS * args.SliceSize;

  // don't reduce this if we cut the slice size in half below, because if that
  // happens, the last chunk will be larger than the other chunks, and we will
  // need the extra buffer space
  args.BufferSliceStride = args.SliceSize + paddingN;

  args.BufferMisalignedN = misalignedN;

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

  args.ThisPtrToNextOutput = (T**)&(comm->ptrs[nextId].local->recvPtrs[0]);
  args.PrevPtrToThisOutput = (T**)&(comm->ptrs[prevId].remote->recvPtrs[0]);

  args.ThisInput = (const T*)sendbuff;
  args.ThisOutput = (volatile T*)recvbuff;
  args.ThisBuffer = (volatile T*)comm->ptrs[prevId].local->buff;
  args.NextBuffer = (volatile T*)comm->ptrs[nextId].remote->buff;

  args.ThisNewDataAvailableFlag = comm->ptrs[prevId].local->flags;
  args.NextNewDataAvailableFlag = comm->ptrs[nextId].remote->flags;
  args.ThisChunkDoneFlag = comm->ptrs[nextId].local->flags + 1;
  args.PrevChunkDoneFlag = comm->ptrs[prevId].remote->flags + 1;

  if (comm->nDev == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    if( comm->useRemoteRecv ) {
      AllGatherKernel<NUM_THREADS, UNROLL_COUNT, true, T>
	<<<1, NUM_THREADS + 1, 0, stream>>>(args);
    } else {
      AllGatherKernel<NUM_THREADS, UNROLL_COUNT, false, T>
	<<<1, NUM_THREADS + 1, 0, stream>>>(args);
    }
  }
  return ncclSuccess;
}

class AllGatherFunctor {
public:
  ncclResult_t operator()(const void* sendbuff, void* recvbuff,
      int count, ncclDataType_t datatype, ncclRedOp_t /*dummy operation*/,
      int /*dummy root*/, ncclComm* comm, cudaStream_t stream) {
    int numUnroll = 16; // this is optimal on dt07 with 4 GPUs

    switch (datatype) {
    case ncclChar:
      return ncclAllGatherWithType<char>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
    case ncclInt:
      return ncclAllGatherWithType<int>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
#if CUDART_VERSION >= 7050
    case ncclHalf:
      return ncclAllGatherWithType<half>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
#endif
    case ncclFloat:
      return ncclAllGatherWithType<float>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
    case ncclDouble:
      return ncclAllGatherWithType<double>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
    case ncclInt64:
      return ncclAllGatherWithType<long long>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
    case ncclUint64:
      return ncclAllGatherWithType<unsigned long long>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
    }
    return ncclInvalidType;
  }
};

extern "C" DSOGLOBAL
ncclResult_t ncclAllGather(const void* sendbuff, int count, ncclDataType_t datatype,
    void* recvbuff, ncclComm_t comm, cudaStream_t stream) {
  return enqueue(AllGatherFunctor(), sendbuff, recvbuff, count, datatype,
      ncclSum, 0, comm, stream);
}
