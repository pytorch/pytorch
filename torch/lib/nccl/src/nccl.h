/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_H_
#define NCCL_H_

#include <cuda_runtime.h>

#if CUDART_VERSION >= 7050
#include <cuda_fp16.h>
#define CUDA_HAS_HALF 1
#else
#undef CUDA_HAS_HALF
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to communicator */
typedef struct ncclComm* ncclComm_t;

#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;

/* Error type */
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidDevicePointer    =  4,
               ncclInvalidRank             =  5,
               ncclUnsupportedDeviceCount  =  6,
               ncclDeviceNotFound          =  7,
               ncclInvalidDeviceIndex      =  8,
               ncclLibWrapperNotSet        =  9,
               ncclCudaMallocFailed        = 10,
               ncclRankMismatch            = 11,
               ncclInvalidArgument         = 12,
               ncclInvalidType             = 13,
               ncclInvalidOperation        = 14,
               nccl_NUM_RESULTS            = 15 } ncclResult_t;

/* Generates a unique Id with each call. Used to generate commId for
 * ncclCommInitAll. uniqueId will be created in such a way that it is
 * guaranteed to be unique across the host. */
ncclResult_t  ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t pncclGetUniqueId(ncclUniqueId* uniqueId);

/* Creates a new communicator (multi process version).
 * rank must be between 0 and ndev-1 and unique within a communicator clique.
 * ndev is number of logical devices
 * The communicator is created on the current CUDA device.
 * ncclCommInitRank implicitly syncronizes with other ranks, so INIT OF EACH RANK MUST
 * BE CALLED IN A SEPARATE HOST THREADS to avoid deadlock. */
ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int ndev, ncclUniqueId commId, int rank);
ncclResult_t pncclCommInitRank(ncclComm_t* comm, int ndev, ncclUniqueId commId, int rank);

/* Creates a clique of communicators.
 * This is a convenience function to create a single-process communicator clique.
 * Returns an array of ndev newly initialized communicators in comm.
 * comm should be pre-allocated with size at least ndev*sizeof(ncclComm_t).
 * If devlist is NULL, the first ndev CUDA devices are used.
 * Order of devlist defines user-order of processors within the communicator. */
ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
ncclResult_t pncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);

/* Frees resources associated with communicator object. */
void  ncclCommDestroy(ncclComm_t comm);
void pncclCommDestroy(ncclComm_t comm);

/* Returns nice error message. */
const char*  ncclGetErrorString(ncclResult_t result);
const char* pncclGetErrorString(ncclResult_t result);

/* Sets count to number of devices in the communicator clique. */
ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t pncclCommCount(const ncclComm_t comm, int* count);

/* Returns cuda device number associated with communicator. */
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device);
ncclResult_t pncclCommCuDevice(const ncclComm_t comm, int* device);

/* Returns user-ordered "rank" assocaiated with communicator. */
ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank);
ncclResult_t pncclCommUserRank(const ncclComm_t comm, int* rank);

/* Reduction opperation selector */
typedef enum { ncclSum        = 0,
               ncclProd       = 1,
               ncclMax        = 2,
               ncclMin        = 3,
               nccl_NUM_OPS   = 4 } ncclRedOp_t;

/* Data types */
typedef enum { ncclChar       = 0,
               ncclInt        = 1,
#ifdef CUDA_HAS_HALF
               ncclHalf       = 2,
#endif
               ncclFloat      = 3,
               ncclDouble     = 4,
               ncclInt64      = 5,
               ncclUint64     = 6,
               nccl_NUM_TYPES = 7 } ncclDataType_t;

/* Reduces data arrays of length count in sendbuff into recvbuf using op operation.
 * recvbuf may be NULL on all calls except for root device.
 * On the root device, sendbuff and recvbuff are assumed to reside on
 * the same device.
 * Must be called separately for each communicator in communicator clique.
*/
ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuf, int count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclReduce(const void* sendbuff, void* recvbuf, int count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

/* Reduces data arrays of length count in sendbuff using op operation, and leaves
 * identical copies of result on each GPUs recvbuff.
 * Sendbuff and recvbuff are assumed to reside on the same device.
 * Must be called separately for each communicator in communicator clique. */
ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclAllReduce(const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

/* Reduces data in sendbuff using op operation and leaves reduced result scattered
 * over the devices so that recvbuff on the i-th GPU will contain the i-th block of
 * the result. Sendbuff and recvbuff are assumed to reside on same device. Assumes
 * sendbuff has size at least ndev*recvcount elements, where ndev is number of
 * communicators in communicator clique
 * Must be called separately for each communicator in communicator clique.*/
ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff,
    int recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t pncclReduceScatter(const void* sendbuff, void* recvbuff,
    int recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream);

/* Copies count values from root to all other devices.
 * Root specifies the source device in user-order
 * (see ncclCommInit).
 * Must be called separately for each communicator in communicator clique. */
ncclResult_t  ncclBcast(void* buff, int count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclBcast(void* buff, int count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);


/* Each device gathers count values from other GPUs.
 * Result is ordered by comm's logical device order.
 * Assumes recvbuff has size at least ndev*count, where ndev is number of communicators
 * in communicator clique.
 * Sendbuff and recvbuff are assumed to reside on same device.
 * Must be called separately for each communicator in communicator clique. */
ncclResult_t  ncclAllGather(const void* sendbuff, int count, ncclDataType_t datatype,
    void* recvbuff, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclAllGather(const void* sendbuff, int count, ncclDataType_t datatype,
    void* recvbuff, ncclComm_t comm, cudaStream_t stream);


/* The following collective operations are not implemented yet */
///* Gather count values from each device to recvbuff.
// * Result is ordered by comm's logical device order.
// * recvbuff may be NULL for all calls except for root device.
// * On the root device, sendbuff and recvbuff are assumed to reside on the same device.
// * Must be called separately for each communicator in communicator clique. */
// * All GPUs, including root, perform copies into recvbuff.
//ncclResult_t  ncclGather(const void* sendbuff, int count, ncclDataType_t datatype,
//    void* recvbuff, int root, ncclComm_t comm, cudaStream_t stream);
//ncclResult_t pncclGather(const void* sendbuff, int count, ncclDataType_t datatype,
//                        void* recvbuff, int root, ncclComm_t comm, cudaStream_t stream);

///* Root device scatters count values to each devices.
// * sendbuff may be NULL on all devices except a single root
// * device where it is assumed to have size at least nGPUs*count.
// * recvbuff allocated on each gpu, including root, size=count.
// * Result is ordered by comm's logical device order.
// * Called separately for each device in the ncclComm. */
//ncclResult_t  ncclScatter(void* sendbuff, ncclDataType_t datatype, void* recvbuff,
//    int count, int root, ncclComm_t comm, cudaStream_t stream);
//ncclResult_t pncclScatter(void* sendbuff, ncclDataType_t datatype, void* recvbuff,
//    int count, int root, ncclComm_t comm, cudaStream_t stream);
//
///* All GPUs scatter blocks of count elements to other devices.
// * Must be called separately for each device in the ncclComm.
// * sendbuff and recvbuff assumed to reside on same device and
// * have size at least nGPUs*count.
// * Called separately for each device in the ncclComm. */
//ncclResult_t  ncclAllToAll(void* sendbuff, int count, ncclDataType_t datatype,
//    void* recvbuff, ncclComm_t comm, cudaStream_t stream);
//ncclResult_t pncclAllToAll(void* sendbuff, int count, ncclDataType_t datatype,
//    void* recvbuff, ncclComm_t comm, cudaStream_t stream);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard

