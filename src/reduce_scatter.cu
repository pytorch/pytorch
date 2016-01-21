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

#include <cassert>

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

/*
 * numGPUs BLOCKs consisting of recvcount words each
 * BLOCK is split up into NumChunks CHUNKs
 * CHUNK is split up into NUM_SUBCHUNKS SUBCHUNKs
 * SUBCHUNK consists of exactly one SLICE
 * SLICE is most efficiently processed in multiples of UNROLL_SIZE
 *
 * The algorithm has numGPUs steps and each step processes a SLICE (i.e.
 * SUBCHUNK) of a different BLOCK. Only data of the BLOCKs not resident on the
 * GPU need to be communicated, hence (numGPUs - 1) BLOCKs. So the buffer needs
 * to have room for (numGPUs - 1) SLICEs.
 */


// do not encode the subchunk number into the flag, because there is a separate
// flag for each subchunk

// If this is called with STEP, it means that we just finished processing the
// data for step STEP on this GPU, which is the data required on the next GPU
// for step STEP + 1, so we signal the next GPU that its data for step STEP + 1
// is available. This is called by one particular consumer warp and so we select
// the first thread in the warp to set the flag.
#define SIGNAL_NEW_DATA_AVAILABLE(chunk, subchunk, step)                        \
    do {                                                                        \
      args.NextNewDataAvailableFlag[0] =                                        \
          2*((chunk) * args.NumGPUs + (step)) + subchunk + 1;                   \
    } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
// all threads synchronize after thread 0 is done spinning.
#define WAIT_FOR_NEW_DATA(chunk, subchunk, step)                                \
    do {                                                                        \
      if (tid == 0) {                                                           \
        Wait([=] {                                                              \
          return ((volatile int *)args.ThisNewDataAvailableFlag)[0] >=          \
              2*((chunk) * args.NumGPUs + (step)) + subchunk - 1;               \
        });                                                                     \
      }                                                                         \
      BAR(sync, 1, NUM_THREADS);                                                \
    } while (0)

// If this is called with CHUNK, it means that this GPU has just finished
// processing the chunk CHUNK and so the previous GPU can start with CHUNK + 1
#define SIGNAL_CHUNK_DONE(chunk, subchunk)                                      \
    do {                                                                        \
      args.PrevChunkDoneFlag[0] = 2*(chunk) + subchunk + 1;                     \
    } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
// all threads synchronize after thread 0 is done spinning.
#define WAIT_FOR_CHUNK(chunk, subchunk)                                       \
    do {                                                                      \
      if (tid == 0) {                                                         \
        Wait([=] {                                                            \
          return ((volatile int *)args.ThisChunkDoneFlag)[0] >=               \
              2*(chunk) + subchunk - 1;                                       \
        });                                                                   \
      }                                                                       \
      BAR(sync, 1, NUM_THREADS);                                              \
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

/*  if (threadIdx.x == 0)
    printf("[sliceSize=%d] slice=%d numSlices=%d "
        "numBigSlices=%d numSmallSlices=%d bigSliceN=%d smallSliceN=%d "
        "lastSliceN=%d\n", *sliceSize, slice, numSlices, numBigSlices,
        numSmallSlices, bigSliceN, smallSliceN, lastSliceN);
*/
}

template<typename T>
struct ReduceScatterKernelArgs {
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
  return userFromRing[(numGPUs + index - 1 - step) % numGPUs];
}

template<int THREADS, int UNROLL, class FUNC, typename T>
__global__ void ReduceScatterKernel(const ReduceScatterKernelArgs<T> args) {
  if (args.N == 0) return;
  int tid = threadIdx.x;

  // First wait for args.PrevPtrToThisOutput to become nullptr to ensure that
  // the previous GPU is done with a previous collective operation.
  if (tid == 0) {
    Wait([=] {
      return *((T * volatile *)args.PrevPtrToThisOutput) == nullptr; // Wait for previous processor to be done
    });

    *((T * volatile *)args.PrevPtrToThisOutput) = (T*)args.ThisOutput; // Tell Previous I'm starting
    Wait([=] {
      return *((T * volatile *)args.ThisPtrToNextOutput) != nullptr;  // Wait till I've been told next started
    });
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

    // step 0: push data to next GPU
    int step = 0;
    int block = GetBlock(args.ThisId, step, args.UserFromRing, args.NumGPUs);
    int blockOffset = chunkOffset + block * args.N;
    int bufferOffset = block * NUM_SUBCHUNKS * args.BufferSliceStride +
        ((block * args.BufferMisalignedN) % alignof(PackType));
    int sliceSize;

    if (tid < NUM_THREADS) {
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        getSliceSizeAndChunkSize(&sliceSize, s, numSlices, numBigSlices,
            numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

        WAIT_FOR_CHUNK(chunk, s);
        Copy<UNROLL, THREADS>(
            args.NextBuffer + bufferOffset,
            args.ThisInput + blockOffset,
            sliceSize);
        __syncthreads();
        bufferOffset += sliceSize;
        blockOffset += sliceSize;
      }
    } else { // Is consumer
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
      }
    }

    // steps j with 0 < j < k - 1, where k = number of GPUs: reduce and copy to
    // next GPU
    for (step = 1; step < args.NumGPUs - 1; ++step) {
      int block = GetBlock(args.ThisId, step, args.UserFromRing, args.NumGPUs);
      int blockOffset = chunkOffset + block * args.N;
      int bufferOffset = block * NUM_SUBCHUNKS * args.BufferSliceStride +
          ((block * args.BufferMisalignedN) % alignof(PackType));

      if (tid < NUM_THREADS) {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
            getSliceSizeAndChunkSize(&sliceSize, s, numSlices, numBigSlices,
                numSmallSlices, bigSliceN, smallSliceN, lastSliceN);
          WAIT_FOR_NEW_DATA(chunk, s, step);
          Reduce<UNROLL, THREADS, FUNC>(
              args.NextBuffer + bufferOffset,
              args.ThisBuffer + bufferOffset,
              args.ThisInput + blockOffset,
              sliceSize);
          __syncthreads();
          bufferOffset += sliceSize;
          blockOffset += sliceSize;
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
    block = GetBlock(args.ThisId, step, args.UserFromRing, args.NumGPUs);
    blockOffset = chunkOffset + block * args.N;
    bufferOffset = block * NUM_SUBCHUNKS * args.BufferSliceStride +
        ((block * args.BufferMisalignedN) % alignof(PackType));

    if (tid < NUM_THREADS) {
      int outputOffset = 0;
      for (int s=0; s<NUM_SUBCHUNKS; ++s) {
        getSliceSizeAndChunkSize(&sliceSize, s, numSlices, numBigSlices,
            numSmallSlices, bigSliceN, smallSliceN, lastSliceN);
        WAIT_FOR_NEW_DATA(chunk, s, step);
        Reduce<UNROLL, THREADS, FUNC>(
            args.ThisOutput + (chunkOffset + outputOffset),
            args.ThisBuffer + bufferOffset,
            args.ThisInput + blockOffset,
            sliceSize);
        __syncthreads();
        outputOffset += sliceSize;
        bufferOffset += sliceSize;
        blockOffset += sliceSize;
      }
    } else {
      for (int s=0; s<NUM_SUBCHUNKS; ++s) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);

        // signal that chunk is done if this is not the last chunk
        if (chunk + 1 < args.NumChunks) {
          SIGNAL_CHUNK_DONE(chunk, s);
        }
      }
    }
  }

  // wait for the last data to be pushed to us
  if (tid < NUM_THREADS) {
    WAIT_FOR_NEW_DATA(args.NumChunks, NUM_SUBCHUNKS-1, 0);

    if (tid == 0) {
      args.ThisNewDataAvailableFlag[tid] = 0;
      args.ThisChunkDoneFlag[tid] = 0;
      *args.ThisPtrToNextOutput = nullptr;
    }
  }
}

template<class FUNC, typename T>
ncclResult_t ncclReduceScatterWithTypeAndFunc(const void* sendbuff,
    void* recvbuff, const int recvcount, ncclComm* comm, cudaStream_t stream) {
  if (recvcount == 0) {
    return ncclSuccess;
  }
  int index = comm->ncclId;

  int blockSizeInBytes = recvcount * sizeof(T);
  int misalignedBytes = blockSizeInBytes % alignof(uint64_t);

  assert((int)((misalignedBytes / sizeof(T)) * sizeof(T)) == misalignedBytes);

  int misalignedN = misalignedBytes / sizeof(T);
  assert(misalignedN < (int)(sizeof(uint64_t) / sizeof(T)));

  int paddingN = (misalignedN > 0) ? sizeof(uint64_t) / sizeof(T) : 0;

  // There is one slice per GPU, so a slice can be at most bufferN / numGPUs,
  // where bufferN is the number of elements of type T that fit into the buffer.
  // For efficiency, we want the slice size to be a multiple of UNROLL_SIZE
  int bufferN = comm->buffSize / sizeof(T);
  // we only need buffer for k slices and k*k paddings (we need k paddings per
  // block and we have k blocks)
  int bufferNPerSlice = (bufferN - NUM_SUBCHUNKS * comm->nDev * paddingN) /
      (NUM_SUBCHUNKS * comm->nDev);
  int sliceSize = (bufferNPerSlice / UNROLL_SIZE) * UNROLL_SIZE;

  int nextId = (index + 1) % comm->nDev;
  int prevId = (index + comm->nDev - 1) % comm->nDev;

  ReduceScatterKernelArgs<T> args;

  args.ThisId = index;
  args.NumGPUs = comm->nDev;
  args.N = recvcount;

  /* Block j must end up in recvbuff[j], which lives on device with logical
   * index comm->ringFromUser[j]. But the block ordering does not necessarily
   * follow the ring ordering. Hence the order in which a particular GPU
   * processes the different blocks (the correspondence between the step in
   * the reduction algorithm and the block on which a GPU operates in that
   * particular step) is not the same as the ring order.
   *
   * Say we have 4 GPUs and comm->userFromRing = { 1, 2, 0, 3 }. Then there are 4
   * step in the reduction algorithm and block 0 needs to end up device 2,
   * block 1 on device 0, block 2 on device 1, and block 3 needs to end up on
   * device 3. In the last step of the algorithm, each GPU must be processing
   * the block that will end up on that GPU. The blocks that a GPU has to
   * process in the previous steps is determined by the next step because each
   * GPU only hands off data to the next GPU in the ring.
   *
   * In the above example, we get the following table of which block is
   * processed by each GPU in a given step. The columns correspond to the
   * different GPUs while the rows are the steps in the algorithm.
   *
   *      GPU 0   1   2   3
   * step
   *    0     3   1   2   0
   *    1     0   3   1   2
   *    2     2   0   3   1
   *    3     1   2   0   3
   *
   * We note the the rows in the above table are just comm->userFromRing in the last
   * step and the list is cyclicly permuted to the left for each previous
   * step. The columns, which are what the individual GPUs need to know, are
   * comm->userFromRing traversed backwards and starting at index k-1 for GPU k.
   * These columns are what we put into args.BlockVsStep to tell the GPU which
   * block it needs to be processing at a particular step. */
  args.UserFromRing = comm->devUserFromRing;

  args.SliceSize = sliceSize;
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

  // we need 2 * NUM_SUBCHUNKS flags, so use the first NUM_SUBCHUNKS flags
  // to signal the next GPU that new data is available and the following
  // NUM_SUBCHUNKS to signal the previous GPU that a chunk is finished
  args.ThisNewDataAvailableFlag = comm->ptrs[prevId].local->flags;
  args.NextNewDataAvailableFlag = comm->ptrs[nextId].remote->flags;
  args.ThisChunkDoneFlag = comm->ptrs[nextId].local->flags + 1;
  args.PrevChunkDoneFlag = comm->ptrs[prevId].remote->flags + 1;

  ReduceScatterKernel<NUM_THREADS, UNROLL_COUNT, FUNC, T>
      <<<1, NUM_THREADS + 1, 0, stream>>>(args);
  return ncclSuccess;
}

template<typename T>
ncclResult_t ncclReduceScatterWithType(const void* sendbuff, void* recvbuff,
    int recvcount, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  switch (op) {
  case ncclSum:
    return ncclReduceScatterWithTypeAndFunc<FuncSum<T>, T>(
        sendbuff, recvbuff, recvcount, comm, stream);
  case ncclProd:
    return ncclReduceScatterWithTypeAndFunc<FuncProd<T>, T>(
        sendbuff, recvbuff, recvcount, comm, stream);
  case ncclMax:
    return ncclReduceScatterWithTypeAndFunc<FuncMax<T>, T>(
        sendbuff, recvbuff, recvcount, comm, stream);
  case ncclMin:
    return ncclReduceScatterWithTypeAndFunc<FuncMin<T>, T>(
        sendbuff, recvbuff, recvcount, comm, stream);
  }
  return ncclInvalidOperation;
}

class ReduceScatterFunctor {
public:
  ncclResult_t operator()(const void* sendbuff, void* recvbuff,
      int recvcount, ncclDataType_t datatype, ncclRedOp_t op, int /*root*/,
      ncclComm* comm, cudaStream_t stream) {

    switch (datatype) {
    case ncclChar:
      return ncclReduceScatterWithType<char>(sendbuff, recvbuff, recvcount,
          op, comm, stream);
    case ncclInt:
      return ncclReduceScatterWithType<int>(sendbuff, recvbuff, recvcount,
          op, comm, stream);
#ifdef CUDA_HAS_HALF
    case ncclHalf:
      return ncclReduceScatterWithType<half>(sendbuff, recvbuff, recvcount,
          op, comm, stream);
#endif
    case ncclFloat:
      return ncclReduceScatterWithType<float>(sendbuff, recvbuff, recvcount,
          op, comm, stream);
    case ncclDouble:
      return ncclReduceScatterWithType<double>(sendbuff, recvbuff, recvcount,
          op, comm, stream);
    case ncclInt64:
      return ncclReduceScatterWithType<long long>(sendbuff, recvbuff, recvcount,
          op, comm, stream);
    case ncclUint64:
      return ncclReduceScatterWithType<unsigned long long>(sendbuff, recvbuff, recvcount,
          op, comm, stream);
    }
    return ncclInvalidType;
  }
};

extern "C" DSOGLOBAL
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
    int recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
    cudaStream_t stream) {
  return enqueue(ReduceScatterFunctor(), sendbuff, recvbuff, recvcount,
      datatype, op, 0, comm, stream);
}
