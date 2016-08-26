/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#ifndef enqueue_h_
#define enqueue_h_

#include "core.h"

int getRingIndex(const ncclComm_t comm, int device);
void lockEventQueue(EventQueue* eq);
void releaseEventQueue(EventQueue* eq);
void CUDART_CB freeEvent(cudaStream_t stream, cudaError_t status, void* eq_void);

/* Syncronize with user stream and launch the collective.
 * All work is performed asynchronously with the host thread.
 * The actual collective should be a functor with the
 * folloaing signature.
 * ncclResult_t collective(void* sendbuff, void* recvbuff,
 *                         int count, ncclDataType_t type, ncclRedOp_t op,
 *                         int root, ncclComm_t comm);
 * Unneeded arguments should be ignored. The collective may
 * assume that the appropriate cuda device has been set. */
template<typename ColFunc>
ncclResult_t enqueue(ColFunc colfunc,
                     const void* sendbuff,
                     void* recvbuff,
                     int count,
                     ncclDataType_t type,
                     ncclRedOp_t op,
                     int root,
                     ncclComm_t comm,
                     cudaStream_t stream)
{
  int curDevice;
  CUDACHECK( cudaGetDevice(&curDevice) );

  // No need for a mutex here because we assume that all enqueue operations happen in a fixed
  // order on all devices. Thus, thread race conditions SHOULD be impossible.
  EventQueue* eq = &comm->events;

  // Ensure that previous collective is complete
  cudaError_t flag = cudaEventQuery(eq->isDone[eq->back]);
  if( flag == cudaErrorNotReady )
    CUDACHECK( cudaStreamWaitEvent(stream, eq->isDone[eq->back], 0) );

  // Launch the collective here
  ncclResult_t ret = colfunc(sendbuff, recvbuff, count, type, op, root, comm, stream);

  eq->back = (eq->back + 1) % MAXQUEUE;
  CUDACHECK( cudaEventRecord(eq->isDone[eq->back], stream) );
  return ret;
}

#endif // End include guard

