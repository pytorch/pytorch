/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "primitives.h"

#define NUM_SUBSTEPS 2
#define NUM_BUFCHUNKS 2

// Increase Step and poffset/noffset for buffer sync
#define NEXT_STEP \
  step++; \
  poffset = noffset; \
  noffset += sliceSize; \
  if (noffset == buffSize) noffset = 0;

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

template<int THREADS, int UNROLL, class FUNC, typename T>
__launch_bounds__(THREADS+WARP_SIZE, 1)
__global__ void ReduceScatterKernel(const KernelArgs<T> args) {
  const int tid = threadIdx.x;
  __shared__ DevRing<T> ring;

  LoadRing<THREADS>(args.ring, &ring);
  __syncthreads();

  if (tid == 0) {
    WaitFlag prevCommOp(ring.prevOpCounter, 0);
    WaitFlag nextCommOp(ring.nextOpCounter, 0);
    prevCommOp.wait(args.opIndex);
    nextCommOp.wait(args.opIndex);
  }
  __syncthreads();

  WaitFlag waitDoneFromNext(ring.recvFlagFromNext, -NUM_BUFCHUNKS*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring.recvFlagFromPrev, -1*NUM_SUBSTEPS);
  PostFlag postDoneToPrev(ring.sendFlagToPrev, -1*NUM_SUBSTEPS);
  PostFlag postReadyToNext(ring.sendFlagToNext, 0);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T, FUNC> Prims;

  const int size = args.N;
  const int nranks = args.nRanks;
  const int buffSize = args.buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;
  
  int step = 0;
  int poffset, noffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput =  args.ThisOutput;
  T * __restrict__ prevInput = ring.recvBuffer;
  T * __restrict__ nextOutput =  ring.sendBuffer;

  for (int chunkOffset = 0; chunkOffset < size; chunkOffset += sliceSize) {
    /////////////// begin ReduceScatter steps ///////////////
    int offset;
    int maxOffset = size-chunkOffset;
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring.userRank[nranks-1];
    offset = chunkOffset + rankDest * size;

    Prims::Copy(
        thisInput  + offset,
        nextOutput + noffset,
        sliceSize, maxOffset,
        step,
        waitDoneFromNext, waitReadyFromPrev,
        postReadyToNext, postDoneToPrev);

    NEXT_STEP; // Increases step, poffset, noffset

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      rankDest = ring.userRank[nranks-j];
      offset = chunkOffset + rankDest * size;

      Prims::Reduce(
          prevInput  + poffset,
          thisInput  + offset,
          nextOutput + noffset,
          sliceSize, maxOffset,
          step,
          waitDoneFromNext, waitReadyFromPrev,
          postReadyToNext, postDoneToPrev);

      NEXT_STEP;
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    rankDest = ring.userRank[0];
    offset = chunkOffset + rankDest * size;

    Prims::Reduce(
        prevInput  + poffset,
        thisInput  + offset,
        thisOutput + chunkOffset,
        sliceSize, maxOffset,
        step,
        waitDoneFromNext, waitReadyFromPrev,
        postReadyToNext, postDoneToPrev);

    NEXT_STEP;
  }

  // wait for the last data to be pushed to us
  if (tid == 0) {
    // Wait for last update from next then reset the flag
    waitDoneFromNext.wait(NUM_SUBSTEPS*(step+NUM_BUFCHUNKS-1));
    *ring.recvFlagFromNext = 0;

    // Wait for last update from prev then reset the flag
    waitReadyFromPrev.wait(NUM_SUBSTEPS*(step+1));
    *ring.recvFlagFromPrev = 0;

    incrementOpCounter(&args);
  }
}

#define THREADS 512
#define UNROLL 8

template<class FUNC, typename T>
ncclResult_t RingReduceScatter(const void* sendbuff, void* recvbuff,
    const int count, ncclComm* comm, cudaStream_t stream) {
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream), ncclUnhandledCudaError);
  } else {
    KernelArgs<T> args;
    ArgsSetup(&args, sendbuff, recvbuff, 0, count, comm);
    LAUNCH_KERNEL(ReduceScatterKernel, THREADS, UNROLL, FUNC, T, args, stream);
  }

  return ncclSuccess;
}

template<typename T, template <typename> class RedOp>
class ReduceScatter {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      int count, int /*root*/, ncclComm* comm, cudaStream_t stream) {
    return RingReduceScatter<RedOp<T>, T>(sendbuff, recvbuff, count, comm, stream);
  }
};

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, int recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, int recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NCCLCHECK(ArgsCheck(sendbuff, recvbuff, recvcount, datatype, op, 0, comm, "ReduceScatter"));
  return enqueue<ReduceScatter>(sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream);
}

