!*************************************************************************
!* Copyright (c) 2016 Research Computing Services (RCS), University of
!* Cambridge. All rights reserved.
!*
!* See LICENSE.txt for license information
!*************************************************************************

program test
use iso_c_binding
use iso_fortran_env
use cudaFor
use ncclFor
implicit none
integer(int32) :: stat, i
real(real32) :: err
integer(int32) :: nEl, nDev
type(ncclDataType) :: dataType
type(ncclComm), allocatable :: comm(:)
integer(int32), allocatable :: devList(:)
type(ncclResult) :: res
integer(int32) :: cudaDev, rank
integer(cuda_stream_kind), allocatable :: stream(:)
integer(int32) :: time(8)
integer(int32), allocatable :: seed(:)
real(real32), allocatable, target :: hostBuff(:, :)
type(c_ptr), allocatable :: hostBuffPtr(:)
type(c_devptr), allocatable :: sendBuffPtr(:)
type(c_devptr), allocatable :: recvBuffPtr(:)

  nEl = 2621440

!  nDev = 2
  stat = cudaGetDeviceCount(nDev)

  dataType = ncclFloat

  allocate(comm(nDev))
  allocate(devList(nDev))

  do i = 1, nDev
    devList(i) = i - 1
  end do

  res = ncclCommInitAll(comm, nDev, devList)

  do i = 1, nDev
    res = ncclCommCuDevice(comm(i), cudaDev)
    res = ncclCommUserRank(comm(i), rank)
  end do

  allocate(stream(nDev))

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaStreamCreate(stream(i))
  end do

  call date_and_time(values = time)
  call random_seed(size = i)
  allocate(seed(i))
  call random_seed(get = seed)
  seed = 60 * 60 * 1000 * time(5) + 60 * 1000 * time(6) + 1000 * time(7) + time(8) - seed
  call random_seed(put = seed)

  allocate(hostBuff(nEl * nDev, nDev + 1))

  call random_number(hostBuff)

  print "(a)", "before allgather:"
  do i = 1, nDev
    err = maxval(abs(hostBuff(:, i) / hostBuff(:, nDev + 1) - 1.0_real32))
    print "(a, i2.2, a, e11.4e2)", "maximum error of rank ", i - 1, " vs sendbuff = ", err
  end do

  allocate(hostBuffPtr(nDev))

  do i = 1, nDev
    hostBuffPtr(i) = c_loc(hostBuff((i - 1) * nEl + 1, nDev + 1))
  end do

  allocate(sendBuffPtr(nDev))

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaMalloc(sendBuffPtr(i), nEl * c_sizeof(hostBuff(1, 1)))
    stat = cudaMemcpy(sendBuffPtr(i), hostBuffPtr(i), nEl * c_sizeof(hostBuff(1, 1)), cudaMemcpyHostToDevice)
  end do

  do i = 1, nDev
    hostBuffPtr(i) = c_loc(hostBuff(1, i))
  end do

  allocate(recvBuffPtr(nDev))

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaMalloc(recvBuffPtr(i), nEl * c_sizeof(hostBuff(1, 1)) * nDev)
    stat = cudaMemcpy(recvBuffPtr(i), hostBuffPtr(i), nEl * c_sizeof(hostBuff(1, 1)) * nDev, cudaMemcpyHostToDevice)
  end do

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    res = ncclAllGather(sendBuffPtr(i), nEl, dataType, recvBuffPtr(i), comm(i), stream(i))
  end do

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaStreamSynchronize(stream(i))
  end do

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaMemcpy(hostBuffPtr(i), recvBuffPtr(i), nEl * c_sizeof(hostBuff(1, 1)) * nDev, cudaMemcpyDeviceToHost)
  end do

  print "(a)", ""
  print "(a)", "after allgather:"
  do i = 1, nDev
    err = maxval(abs(hostBuff(:, i) / hostBuff(:, nDev + 1) - 1.0_real32))
    print "(a, i2.2, a, e11.4e2)", "maximum error of rank ", i - 1, " vs sendbuff = ", err
  end do

  do i = 1, nDev
    hostBuffPtr(i) = c_loc(hostBuff((i - 1) * nEl + 1, 1))
  end do

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaMemcpy(hostBuffPtr(i), sendBuffPtr(i), nEl * c_sizeof(hostBuff(1, 1)), cudaMemcpyDeviceToHost)
  end do

  err = maxval(abs(hostBuff(:, 1) / hostBuff(:, nDev + 1) - 1.0_real32))
  print "(a)", ""
  print "(a, e11.4e2)", "maximum error in sendbuff = ", err
  print "(a)", ""

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaFree(recvBuffPtr(i))
  end do

  deallocate(recvBuffPtr)

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaFree(sendBuffPtr(i))
  end do

  deallocate(sendBuffPtr)

  deallocate(hostBuffPtr)

  deallocate(hostBuff)

  deallocate(seed)

  do i = 1, nDev
    stat = cudaSetDevice(devList(i))
    stat = cudaStreamDestroy(stream(i))
  end do

  deallocate(stream)

  do i = 1, nDev
    call ncclCommDestroy(comm(i))
  end do

  deallocate(devList)
  deallocate(comm)

end program test
