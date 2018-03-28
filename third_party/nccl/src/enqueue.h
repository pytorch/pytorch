/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef enqueue_h_
#define enqueue_h_

#include "core.h"
#include "reduce_kernel.h"

/* Syncronize previous collective (if in different stream) and enqueue
 * collective. Work is performed asynchronously with the host thread.
 * The ColFunc class should be templated on the datatype and reduction
 * operator (if applicable) and define a static entry() method as
 * follows.
 *   template <typename T, template <typename> class RedOp>
 *   class CollectiveFunctor {
 *     public:
 *     static ncclResult_t entry(const void* sendbuff, void* recvbuff, int count,
 *         int root, ncclComm* comm, cudaStream_t stream);
 *   };
 * The entry() method can assume that the appropriate cuda device has been set. */
template< template<typename, template<typename> class> class ColFunc,
          typename T,
          template<typename> class Op >
ncclResult_t enqueue(const void* sendbuff,
                     void* recvbuff,
                     int count,
                     int root,
                     ncclComm_t comm,
                     cudaStream_t stream)
{
  if (stream != comm->prevStream) { // sync required for calls in different streams
    comm->prevStream = stream;
    CUDACHECK(cudaStreamWaitEvent(stream, comm->doneEvent, 0), ncclUnhandledCudaError);
  }

  ncclResult_t ret;
  ret = ColFunc<T, Op>::entry(sendbuff, recvbuff, count, root, comm, stream);

  // Always have to record done event because we don't know what stream next
  // collective will be in.
  CUDACHECK(cudaEventRecord(comm->doneEvent, stream), ncclUnhandledCudaError);
  comm->opSched += 1;
  return ret;
}


// This version decodes type
template< template<typename, template<typename> class> class ColFunc,
          template<typename> class Op >
ncclResult_t enqueue(const void* sendbuff,
                     void* recvbuff,
                     int count,
                     ncclDataType_t type,
                     int root,
                     ncclComm_t comm,
                     cudaStream_t stream)
{
  switch(type) {
  case ncclChar:
    return enqueue<ColFunc, char, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclInt:
    return enqueue<ColFunc, int, Op>(sendbuff, recvbuff, count, root, comm, stream);
#ifdef CUDA_HAS_HALF
  case ncclHalf:
    return enqueue<ColFunc, half, Op>(sendbuff, recvbuff, count, root, comm, stream);
#endif
  case ncclFloat:
    return enqueue<ColFunc, float, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclDouble:
    return enqueue<ColFunc, double, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclInt64:
    return enqueue<ColFunc, long long, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclUint64:
    return enqueue<ColFunc, unsigned long long, Op>(sendbuff, recvbuff, count, root, comm, stream);
  default:
    WARN("Invalid ncclType %d", type);
    return ncclInvalidType;
  }
}

// This version decodes both type and reduction op
template< template<typename, template<typename> class> class ColFunc>
ncclResult_t enqueue(const void* sendbuff,
                     void* recvbuff,
                     int count,
                     ncclDataType_t type,
                     ncclRedOp_t op,
                     int root,
                     ncclComm_t comm,
                     cudaStream_t stream)
{
  switch(op) {
  case ncclSum:
    return enqueue<ColFunc, FuncSum>(sendbuff, recvbuff, count, type, root, comm, stream);
  case ncclProd:
    return enqueue<ColFunc, FuncProd>(sendbuff, recvbuff, count, type, root, comm, stream);
  case ncclMax:
    return enqueue<ColFunc, FuncMax>(sendbuff, recvbuff, count, type, root, comm, stream);
  case ncclMin:
    return enqueue<ColFunc, FuncMin>(sendbuff, recvbuff, count, type, root, comm, stream);
  default:
    WARN("Invalid ncclRedOp: %d", op);
    return ncclInvalidOperation;
  }
}

#endif // End include guard

