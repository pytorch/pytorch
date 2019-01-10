!*************************************************************************
!* Copyright (c) 2016 Research Computing Services (RCS), University of
!* Cambridge. All rights reserved.
!*
!* See LICENSE.txt for license information
!*************************************************************************

#ifndef _CUDA

!Start cudaFor module
module cudaFor
use iso_c_binding
implicit none
private
public :: c_devptr
public :: cudaMemcpyKind,           &
          cudaMemcpyHostToHost,     &
          cudaMemcpyHostToDevice,   &
          cudaMemcpyDeviceToHost,   &
          cudaMemcpyDeviceToDevice, &
          cudaMemcpyDefault
public :: cuda_stream_kind
public :: cudaGetDeviceCount
public :: cudaSetDevice
public :: cudaMalloc
public :: cudaMemcpy
public :: cudaFree
public :: cudaStreamCreate
public :: cudaStreamSynchronize
public :: cudaStreamDestroy

!Start types

!Start c_devptr
type, bind(c) :: c_devptr
type(c_ptr) :: member
end type c_devptr
!End c_devptr

!Start cudaMemcpyKind
type, bind(c) :: cudaMemcpyKind
integer(c_int) :: member
end type cudaMemcpyKind

type(cudaMemcpyKind), parameter :: cudaMemcpyHostToHost     = cudaMemcpyKind(0), &
                                   cudaMemcpyHostToDevice   = cudaMemcpyKind(1), &
                                   cudaMemcpyDeviceToHost   = cudaMemcpyKind(2), &
                                   cudaMemcpyDeviceToDevice = cudaMemcpyKind(3), &
                                   cudaMemcpyDefault        = cudaMemcpyKind(4)
!End cudaMemcpyKind

!Start cuda_stream_kind
integer(c_intptr_t), parameter :: cuda_stream_kind = c_intptr_t
!End cuda_stream_kind

!End types

!Start interfaces

!Start cudaGetDeviceCount
interface cudaGetDeviceCount
integer(c_int) function cudaGetDeviceCount(count) bind(c, name = "cudaGetDeviceCount")
import :: c_int
implicit none
integer(c_int) :: count
end function cudaGetDeviceCount
end interface cudaGetDeviceCount
!End cudaGetDeviceCount

!Start cudaSetDevice
interface cudaSetDevice
integer(c_int) function cudaSetDevice(device) bind(c, name = "cudaSetDevice")
import :: c_int
implicit none
integer(c_int), value :: device
end function cudaSetDevice
end interface cudaSetDevice
!End cudaSetDevice

!Start cudaMalloc
interface cudaMalloc
integer(c_int) function cudaMalloc(devPtr, size) bind(c, name = "cudaMalloc")
import :: c_int, c_size_t
import :: c_devptr
implicit none
type(c_devptr) :: devPtr
integer(c_size_t), value :: size
end function cudaMalloc
end interface cudaMalloc
!End cudaMalloc

!Start cudaMemcpy
interface cudaMemcpy

!Start cudaMemcpyH2D
integer(c_int) function cudaMemcpyH2D(dst, src, count, kind) bind(c, name = "cudaMemcpy")
import :: c_ptr, c_int, c_size_t
import :: c_devptr, cudaMemcpyKind
implicit none
type(c_devptr), value :: dst
type(c_ptr), value :: src
integer(c_size_t), value :: count
type(cudaMemcpyKind), value :: kind
end function cudaMemcpyH2D
!End cudaMemcpyH2D

!Start cudaMemcpyD2H
integer(c_int) function cudaMemcpyD2H(dst, src, count, kind) bind(c, name = "cudaMemcpy")
import :: c_ptr, c_int, c_size_t
import :: c_devptr, cudaMemcpyKind
implicit none
type(c_ptr), value :: dst
type(c_devptr), value :: src
integer(c_size_t), value :: count
type(cudaMemcpyKind), value :: kind
end function cudaMemcpyD2H
!End cudaMemcpyD2H

end interface cudaMemcpy
!End cudaMemcpy

!Start cudaFree
interface cudaFree
integer(c_int) function cudaFree(devPtr) bind(c, name = "cudaFree")
import :: c_int
import :: c_devptr
implicit none
type(c_devptr), value :: devPtr
end function cudaFree
end interface cudaFree
!End cudaFree

!Start cudaStreamCreate
interface cudaStreamCreate
integer(c_int) function cudaStreamCreate(pStream) bind(c, name = "cudaStreamCreate")
import :: c_int
import :: cuda_stream_kind
implicit none
integer(cuda_stream_kind) :: pStream
end function cudaStreamCreate
end interface cudaStreamCreate
!End cudaStreamCreate

!Start cudaStreamSynchronize
interface cudaStreamSynchronize
integer(c_int) function cudaStreamSynchronize(stream) bind(c, name = "cudaStreamSynchronize")
import :: c_int
import :: cuda_stream_kind
implicit none
integer(cuda_stream_kind), value :: stream
end function cudaStreamSynchronize
end interface cudaStreamSynchronize
!End cudaStreamSynchronize

!Start cudaStreamDestroy
interface cudaStreamDestroy
integer(c_int) function cudaStreamDestroy(stream) bind(c, name = "cudaStreamDestroy")
import :: c_int
import :: cuda_stream_kind
implicit none
integer(cuda_stream_kind), value :: stream
end function cudaStreamDestroy
end interface cudaStreamDestroy
!End cudaStreamDestroy

!End interfaces

end module cudaFor
!End cudaFor module

#endif
