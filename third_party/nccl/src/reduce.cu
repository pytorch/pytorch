/*************************************************************************
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "primitives.h"

#define NUM_SUBSTEPS 2
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
__global__ void ReduceKernel(const KernelArgs<T> args) {
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

  WaitFlag waitDoneFromNext(ring.recvFlagFromNext, (1-NUM_BUFCHUNKS)*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring.recvFlagFromPrev, 0);
  PostFlag postDoneToPrev(ring.sendFlagToPrev, 0);
  PostFlag postReadyToNext(ring.sendFlagToNext, 0);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T, FUNC> Prims;

  const int size = args.N;
  const int nranks = args.nRanks;
  const int rank = ring.userRank[0];
  const int prevRank = ring.userRank[nranks-1];
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
    if (prevRank == root) {
      Prims::Copy(
          thisInput + offset,
          nextOutput + boffset,
          sliceSize, maxOffset,
          step,
          waitDoneFromNext,
          postReadyToNext);
    } else if (rank == root) {
      Prims::Reduce(
          thisInput + offset,
          prevInput + boffset,
          thisOutput + offset,
          sliceSize, maxOffset,
          step,
          waitReadyFromPrev,
          postDoneToPrev);
    } else {
      Prims::Reduce(
          thisInput + offset,
          prevInput + boffset,
          nextOutput + boffset,
          sliceSize, maxOffset,
          step,
          waitDoneFromNext, waitReadyFromPrev,
          postReadyToNext, postDoneToPrev);
    }
    NEXT_STEP; // Increases step, boffset
  }

  // wait for the last data to be pushed to us
  if (tid == 0) {
    if (rank != root) {
      // Wait for last update from next then reset the flag
      waitDoneFromNext.wait(NUM_SUBSTEPS*(step+NUM_BUFCHUNKS-1));
      *ring.recvFlagFromNext = 0;
    }

    if (prevRank != root) {
      // reset the flag
      *ring.recvFlagFromPrev = 0;
    }

    incrementOpCounter(&args);
  }
}

#define THREADS 512
#define UNROLL 8

template<class FUNC, typename T>
ncclResult_t RingReduce(const void* sendbuff, void* recvbuff, const int count, const int root,
    ncclComm* comm, cudaStream_t stream) {
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream), ncclUnhandledCudaError);
  } else {
    KernelArgs<T> args;
    ArgsSetup(&args, sendbuff, recvbuff, root, count, comm);
    LAUNCH_KERNEL(ReduceKernel, THREADS, UNROLL, FUNC, T, args, stream);
  }

  return ncclSuccess;
}

template<typename T, template<typename> class RedOp>
class ReduceFunctor {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      int count, int root, ncclComm* comm, cudaStream_t stream) {
    return RingReduce<RedOp<T>, T>(sendbuff, recvbuff, count, root, comm, stream);
  }
};

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ArgsCheck(sendbuff, recvbuff, count, datatype, op, root, comm, "Reduce"));
  return enqueue<ReduceFunctor>(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

