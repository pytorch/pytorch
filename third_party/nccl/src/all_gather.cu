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
__global__ void AllGatherKernel(const KernelArgs<T> args) {
  const int tid = threadIdx.x;
  __shared__ T* sharedNextOutput;
  __shared__ DevRing<T> ring;
  bool pushrecv = args.pushrecv;

  LoadRing<THREADS>(args.ring, &ring);
  __syncthreads();

  if (tid == 0) {
    WaitFlag prevCommOp(ring.prevOpCounter, 0);
    WaitFlag nextCommOp(ring.nextOpCounter, 0);
    prevCommOp.wait(args.opIndex);
    nextCommOp.wait(args.opIndex);
    if (pushrecv) {
      *ring.sendPtrToPrev = (T*)args.ThisOutput;
      Wait([=] {
        return *ring.recvPtrFromNext != nullptr;
      });
      sharedNextOutput = *ring.recvPtrFromNext;
      *ring.recvPtrFromNext = nullptr;
    }
  }
  __syncthreads();

  WaitFlag waitDoneFromNext(ring.recvFlagFromNext, -NUM_BUFCHUNKS*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring.recvFlagFromPrev, -1*NUM_SUBSTEPS);
  PostFlag postDoneToPrev(ring.sendFlagToPrev, -1*NUM_SUBSTEPS);
  PostFlag postReadyToNext(ring.sendFlagToNext, 0);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T> Prims;

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
    /////////////// begin AllGather steps ///////////////
    int offset;
    int maxOffset = size-chunkOffset;
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring.userRank[0];
    offset = chunkOffset + rankDest * size;

    if (thisInput == thisOutput) {
      Prims::Copy(
          thisInput  + offset,
          pushrecv ? sharedNextOutput + offset : nextOutput + noffset,
          sliceSize, maxOffset,
          step,
          waitDoneFromNext, waitReadyFromPrev,
          postReadyToNext, postDoneToPrev);
    } else {
      Prims::DoubleCopy(
          thisInput  + chunkOffset,
          thisOutput + offset,
          pushrecv ? sharedNextOutput + offset : nextOutput + noffset,
          sliceSize, maxOffset,
          step,
          waitDoneFromNext, waitReadyFromPrev,
          postReadyToNext, postDoneToPrev);
    }

    NEXT_STEP; // Increases step, poffset, noffset

    // k-2 steps: copy to next GPU
    if (pushrecv) {
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ring.userRank[nranks-j];
        offset = chunkOffset + rankDest * size;

        Prims::Copy(
            thisOutput + offset,
            sharedNextOutput + offset,
            sliceSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);

        NEXT_STEP;
      }
    } else {
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ring.userRank[nranks-j];
        offset = chunkOffset + rankDest * size;

        Prims::DoubleCopy(
            prevInput + poffset,
            thisOutput + offset,
            nextOutput + noffset,
            sliceSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);

        NEXT_STEP;
      }

      // Make final copy from buffer to dest.
      rankDest = ring.userRank[1];
      offset = chunkOffset + rankDest * size;

      // Here we need to copy from buffer to this output.
      Prims::Copy(
          prevInput + poffset,
          thisOutput + offset,
          sliceSize, maxOffset,
          step,
          waitDoneFromNext, waitReadyFromPrev,
          postReadyToNext, postDoneToPrev);

      NEXT_STEP;
    }
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
ncclResult_t RingAllGather(const void* sendbuff, void* recvbuff,
    const int count, ncclComm* comm, cudaStream_t stream) {
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream), ncclUnhandledCudaError);
  } else {
    KernelArgs<T> args;
    ArgsSetup(&args, sendbuff, recvbuff, 0, count, comm);
    LAUNCH_KERNEL(AllGatherKernel, THREADS, UNROLL, FUNC, T, args, stream);
  }

  return ncclSuccess;
}

template<typename T, template<typename> class RedOp>
class AllGather {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      int count, int /*root*/, ncclComm* comm, cudaStream_t stream) {
    return RingAllGather<RedOp<T>, T>(sendbuff, recvbuff, count, comm, stream);
  }
};

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, int count, ncclDataType_t datatype,
    void* recvbuff, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, int count, ncclDataType_t datatype,
    void* recvbuff, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ArgsCheck(sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, "AllGather"));
  return enqueue<AllGather, FuncNull>(sendbuff, recvbuff, count, datatype, 0, comm, stream);
}

