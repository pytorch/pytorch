!*************************************************************************
!* Copyright (c) 2016 Research Computing Services (RCS), University of
!* Cambridge. All rights reserved.
!*
!* See LICENSE.txt for license information
!*************************************************************************

!Start defines
#define NCCL_UNIQUE_ID_BYTES 128
!End defines

!Start ncclFor module
module ncclFor
use iso_c_binding
use cudaFor
implicit none
private
public :: ncclUniqueId
public :: ncclComm
public :: ncclResult,                 &
          ncclSuccess,                &
          ncclUnhandledCudaError,     &
          ncclSystemError,            &
          ncclInternalError,          &
          ncclInvalidDevicePointer,   &
          ncclInvalidRank,            &
          ncclUnsupportedDeviceCount, &
          ncclDeviceNotFound,         &
          ncclInvalidDeviceIndex,     &
          ncclLibWrapperNotSet,       &
          ncclCudaMallocFailed,       &
          ncclRankMismatch,           &
          ncclInvalidArgument,        &
          ncclInvalidType,            &
          ncclInvalidOperation,       &
          nccl_NUM_RESULTS
public :: ncclDataType, &
          ncclChar,     &
          ncclInt,      &
#ifdef CUDA_HAS_HALF
          ncclHalf,     &
#endif
          ncclFloat,    &
          ncclDouble,   &
          ncclInt64,    &
          ncclUInt64,   &
          nccl_NUM_TYPES
public :: ncclRedOp, &
          ncclSum,   &
          ncclProd,  &
          ncclMax,   &
          ncclMin,   &
          nccl_NUM_OPS
public :: ncclGetUniqueId
public :: ncclCommInitRank
public :: ncclCommInitAll
public :: ncclCommCuDevice
public :: ncclCommUserRank
public :: ncclCommCount
public :: ncclCommDestroy
public :: ncclReduce
public :: ncclAllReduce
public :: ncclReduceScatter
public :: ncclBcast
public :: ncclAllGather

!Start types

!Start ncclUniqueId
type, bind(c) :: ncclUniqueId
character(c_char) :: internal(NCCL_UNIQUE_ID_BYTES)
end type ncclUniqueId
!End ncclUniqueId

!Start ncclComm
type, bind(c) :: ncclComm
type(c_ptr) :: member
end type ncclComm
!End ncclComm

!Start ncclResult
type, bind(c) :: ncclResult
integer(c_int) :: member
end type ncclResult

type(ncclResult), parameter :: ncclSuccess                = ncclResult( 0), &
                               ncclUnhandledCudaError     = ncclResult( 1), &
                               ncclSystemError            = ncclResult( 2), &
                               ncclInternalError          = ncclResult( 3), &
                               ncclInvalidDevicePointer   = ncclResult( 4), &
                               ncclInvalidRank            = ncclResult( 5), &
                               ncclUnsupportedDeviceCount = ncclResult( 6), &
                               ncclDeviceNotFound         = ncclResult( 7), &
                               ncclInvalidDeviceIndex     = ncclResult( 8), &
                               ncclLibWrapperNotSet       = ncclResult( 9), &
                               ncclCudaMallocFailed       = ncclResult(10), &
                               ncclRankMismatch           = ncclResult(11), &
                               ncclInvalidArgument        = ncclResult(12), &
                               ncclInvalidType            = ncclResult(13), &
                               ncclInvalidOperation       = ncclResult(14), &
                               nccl_NUM_RESULTS           = ncclResult(15)
!End ncclResult

!Start ncclDataType
type, bind(c) :: ncclDataType
integer(c_int) :: member
end type ncclDataType

type(ncclDataType), parameter :: ncclChar       = ncclDataType(0), &
                                 ncclInt        = ncclDataType(1), &
#ifdef CUDA_HAS_HALF
                                 ncclHalf       = ncclDataType(2), &
#endif
                                 ncclFloat      = ncclDataType(3), &
                                 ncclDouble     = ncclDataType(4), &
                                 ncclInt64      = ncclDataType(5), &
                                 ncclUInt64     = ncclDataType(6), &
                                 nccl_NUM_TYPES = ncclDataType(7)
!End ncclDataType

!Start ncclRedOp
type, bind(c) :: ncclRedOp
integer(c_int) :: member
end type ncclRedOp

type(ncclRedOp), parameter :: ncclSum      = ncclRedOp(0), &
                              ncclProd     = ncclRedOp(1), &
                              ncclMax      = ncclRedOp(2), &
                              ncclMin      = ncclRedOp(3), &
                              nccl_NUM_OPS = ncclRedOp(4)
!End ncclRedOp

!End types

!Start interfaces

!Start ncclGetUniqueId
interface ncclGetUniqueId
type(ncclResult) function ncclGetUniqueId(uniqueId) bind(c, name = 'ncclGetUniqueId')
import :: ncclResult, ncclUniqueId
implicit none
type(ncclUniqueId) :: uniqueId
end function ncclGetUniqueId
end interface ncclGetUniqueId
!End ncclGetUniqueId

!Start ncclCommInitRank
interface ncclCommInitRank
type(ncclResult) function ncclCommInitRank(comm, ndev, commId, rank) bind(c, name = 'ncclCommInitRank')
import :: c_int
import :: ncclResult, ncclUniqueId, ncclComm
implicit none
type(ncclComm) :: comm(*)
integer(c_int), value :: ndev
type(ncclUniqueId), value :: commId
integer(c_int), value :: rank
end function ncclCommInitRank
end interface ncclCommInitRank
!End ncclCommInitRank

!Start ncclCommInitAll
interface ncclCommInitAll
type(ncclResult) function ncclCommInitAll(comm, ndev, devlist) bind(c, name = 'ncclCommInitAll')
import :: c_int
import :: ncclResult, ncclComm
implicit none
type(ncclComm) :: comm(*)
integer(c_int), value :: ndev
integer(c_int) :: devlist(*)
end function ncclCommInitAll
end interface ncclCommInitAll
!End ncclCommInitAll

!Start ncclCommCuDevice
interface ncclCommCuDevice
type(ncclResult) function ncclCommCuDevice(comm, devid) bind(c, name = 'ncclCommCuDevice')
import :: c_int
import :: ncclResult, ncclComm
implicit none
type(ncclComm), value :: comm
integer(c_int) :: devid
end function ncclCommCuDevice
end interface ncclCommCuDevice
!End ncclCommCuDevice

!Start ncclCommUserRank
interface ncclCommUserRank
type(ncclResult) function ncclCommUserRank(comm, rank) bind(c, name = 'ncclCommUserRank')
import :: c_int
import :: ncclResult, ncclComm
implicit none
type(ncclComm), value :: comm
integer(c_int) :: rank
end function ncclCommUserRank
end interface ncclCommUserRank
!End ncclCommUserRank

!Start ncclCommCount
interface ncclCommCount
type(ncclResult) function ncclCommCount(comm, count) bind(c, name = 'ncclCommCount')
import :: c_int
import :: ncclResult, ncclComm
implicit none
type(ncclComm), value :: comm
integer(c_int) :: count
end function ncclCommCount
end interface ncclCommCount
!End ncclCommCount

!Start ncclCommDestroy
interface ncclCommDestroy
subroutine ncclCommDestroy(comm) bind(c, name = 'ncclCommDestroy')
import :: ncclComm
implicit none
type(ncclComm), value :: comm
end subroutine ncclCommDestroy
end interface ncclCommDestroy
!End ncclCommDestroy

!Start ncclReduce
interface ncclReduce
type(ncclResult) function ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream) bind(c, name = 'ncclReduce')
import :: c_int
import :: c_devptr, cuda_stream_kind
import :: ncclResult, ncclComm, ncclDataType, ncclRedOp
implicit none
type(c_devptr), value :: sendbuff
type(c_devptr), value :: recvbuff
integer(c_int), value :: count
type(ncclDataType), value :: datatype
type(ncclRedOp), value :: op
integer(c_int), value :: root
type(ncclComm), value :: comm
integer(cuda_stream_kind), value :: stream
end function ncclReduce
end interface ncclReduce
!End ncclReduce

!Start ncclAllReduce
interface ncclAllReduce
type(ncclResult) function ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream) bind(c, name = 'ncclAllReduce')
import :: c_int
import :: c_devptr, cuda_stream_kind
import :: ncclResult, ncclComm, ncclDataType, ncclRedOp
implicit none
type(c_devptr), value :: sendbuff
type(c_devptr), value :: recvbuff
integer(c_int), value :: count
type(ncclDataType), value :: datatype
type(ncclRedOp), value :: op
type(ncclComm), value :: comm
integer(cuda_stream_kind), value :: stream
end function ncclAllReduce
end interface ncclAllReduce
!End ncclAllReduce

!Start ncclReduceScatter
interface ncclReduceScatter
type(ncclResult) function ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream) bind(c, name = 'ncclReduceScatter')
import :: c_int
import :: c_devptr, cuda_stream_kind
import :: ncclResult, ncclComm, ncclDataType, ncclRedOp
implicit none
type(c_devptr), value :: sendbuff
type(c_devptr), value :: recvbuff
integer(c_int), value :: recvcount
type(ncclDataType), value :: datatype
type(ncclRedOp), value :: op
type(ncclComm), value :: comm
integer(cuda_stream_kind), value :: stream
end function ncclReduceScatter
end interface ncclReduceScatter
!End ncclReduceScatter

!Start ncclBcast
interface ncclBcast
type(ncclResult) function ncclBcast(buff, count, datatype, root, comm, stream) bind(c, name = 'ncclBcast')
import :: c_int
import :: c_devptr, cuda_stream_kind
import :: ncclResult, ncclComm, ncclDataType
implicit none
type(c_devptr), value :: buff
integer(c_int), value :: count
type(ncclDataType), value :: datatype
integer(c_int), value :: root
type(ncclComm), value :: comm
integer(cuda_stream_kind), value :: stream
end function ncclBcast
end interface ncclBcast
!End ncclBcast

!Start ncclAllGather
interface ncclAllGather
type(ncclResult) function ncclAllGather(sendbuff, count, datatype, recvbuff, comm, stream) bind(c, name = 'ncclAllGather')
import :: c_int
import :: c_devptr, cuda_stream_kind
import :: ncclResult, ncclComm, ncclDataType
implicit none
type(c_devptr), value :: sendbuff
integer(c_int), value :: count
type(ncclDataType), value :: datatype
type(c_devptr), value :: recvbuff
type(ncclComm), value :: comm
integer(cuda_stream_kind), value :: stream
end function ncclAllGather
end interface ncclAllGather
!End ncclAllGather

!End interfaces

end module ncclFor
!End nccl module
