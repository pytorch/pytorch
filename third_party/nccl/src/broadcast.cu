/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "primitives.h"

#define NUM_SUBSTEPS 4
#define NUM_BUFCHUNKS 2

// Increase Step and boffset for buffer sync
#define NEXT_STEP \
  step++; \
  boffset += sliceSize; \
  if (boffset == buffSize) boffset = 0;

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

template<int THREADS, int UNROLL, class FUNC, typename T>
__launch_bounds__(THREADS+WARP_SIZE, 1)
__global__ void BroadcastKernel(const KernelArgs<T> args) {
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

  WaitFlag waitDoneFromNext(ring.recvFlagFromNext, (1-NUM_BUFCHUNKS)*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring.recvFlagFromPrev, 0);
  PostFlag postDoneToPrev(ring.sendFlagToPrev, 0);
  PostFlag postReadyToNext(ring.sendFlagToNext, 0);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T> Prims;

  const int size = args.N;
  const int rank = ring.userRank[0];
  const int nextRank = ring.userRank[1];
  const int root = args.root;
  const int buffSize = args.buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;
  
  int step = 0;
  int boffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput =  args.ThisOutput;
  T * __restrict__ prevInput = ring.recvBuffer;
  T * __restrict__ nextOutput =  ring.sendBuffer;

  for (int offset = 0; offset < size; offset += sliceSize) {
    int maxOffset = size-offset;
    if (rank == root) {
      Prims::Copy(
          thisInput + offset,
          pushrecv ? sharedNextOutput + offset : nextOutput + boffset,
          sliceSize, maxOffset,
          step,
          waitDoneFromNext,
          postReadyToNext);
    } else if (nextRank == root) {
      if (pushrecv) maxOffset = 0; // Only wait for signals
      Prims::Copy(
          prevInput  + boffset,
          thisOutput + offset,
          sliceSize, maxOffset,
          step,
          waitReadyFromPrev,
          postDoneToPrev);
    } else {
      if (pushrecv) {
        Prims::Copy(
            thisOutput + offset,
            sharedNextOutput + offset,
            sliceSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);
      } else {
        Prims::DoubleCopy(
            prevInput + boffset,
            thisOutput + offset,
            nextOutput + boffset,
	    sliceSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);
      }
    }
    NEXT_STEP; // Increases step, boffset
  }

  // wait for the last data to be pushed to us
  if (tid == 0) {
    if (nextRank != root) {
      // Wait for last update from next then reset the flag
      waitDoneFromNext.wait(NUM_SUBSTEPS*(step+NUM_BUFCHUNKS-1));
      *ring.recvFlagFromNext = 0;
    }

    if (rank != root) {
      // reset the flag
      *ring.recvFlagFromPrev = 0;
    }

    incrementOpCounter(&args);
  }
}

#define THREADS 256
#define UNROLL 8

template<class FUNC, typename T>
ncclResult_t RingBroadcast(void* buff, const int count, const int root,
    ncclComm* comm, cudaStream_t stream) {
  if (comm->nRanks != 1) {
    KernelArgs<T> args;
    ArgsSetup(&args, buff, buff, root, count, comm);
    LAUNCH_KERNEL(BroadcastKernel, THREADS, UNROLL, FUNC, T, args, stream);
  }

  return ncclSuccess;
}

template<typename T, template<typename> class RedOp>
class Broadcast {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      int count, int root, ncclComm* comm, cudaStream_t stream) {
    return RingBroadcast<RedOp<T>, T>(recvbuff, count, root, comm, stream);
  }
};

NCCL_API(ncclResult_t, ncclBcast, void* buff, int count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, int count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ArgsCheck(buff, buff, count, datatype, ncclSum, root, comm, "Bcast"));
  return enqueue<Broadcast, FuncNull>(nullptr, buff, count, datatype, root, comm, stream);
}

