/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "gloo/algorithm.h"
#include "gloo/common/error.h"
#include "gloo/cuda.h"
#include "gloo/cuda_workspace.h"

namespace gloo {

template <typename T, typename W = CudaHostWorkspace<T> >
class CudaAllreduceHalvingDoubling : public Algorithm {
 public:
  CudaAllreduceHalvingDoubling(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const std::vector<cudaStream_t>& streams  = std::vector<cudaStream_t>(),
      bool pipelineBroadcastAndReduce = false);

  virtual ~CudaAllreduceHalvingDoubling() = default;

  virtual void run() override;

 protected:

  void initBinaryBlocks();
  void devicePointerInit();

  // Both workspace types have their own initialization function.
  template <typename U = W>
  void init(
      typename std::enable_if<std::is_same<U, CudaHostWorkspace<T> >::value,
                              typename U::Pointer>::type* = 0);

  template <typename U = W>
  void init(
      typename std::enable_if<std::is_same<U, CudaDeviceWorkspace<T> >::value,
                              typename U::Pointer>::type* = 0);

  template <typename U = W>
  void initReductionsAndBroadcasts(
      typename std::enable_if<std::is_same<U, CudaHostWorkspace<T> >::value,
                              typename U::Pointer>::type* = 0);

  template <typename U = W>
  void initReductionsAndBroadcasts(
      typename std::enable_if<std::is_same<U, CudaDeviceWorkspace<T> >::value,
                              typename U::Pointer>::type* = 0);

  std::vector<CudaDevicePointer<T> > devicePtrs_;
  std::vector<CudaStream> streams_;
  typename W::Pointer scratch_;

  const int count_;
  const int bytes_;
  const size_t steps_;
  const size_t chunks_;
  const size_t chunkSize_;
  const size_t chunkBytes_;

  const CudaReductionFunction<T>* fn_;

  // offsets into the data buffer from which to send during the reduce-scatter
  // these become the offsets at which the process receives during the allgather
  // indexed by step
  std::vector<size_t> sendOffsets_;

  // offsets at which data is reduced during the reduce-scatter and sent from in
  // the allgather
  std::vector<size_t> recvOffsets_;

  std::vector<std::unique_ptr<transport::Buffer>> sendDataBufs_;
  std::vector<std::unique_ptr<transport::Buffer>> recvDataBufs_;

  std::unique_ptr<transport::Buffer> smallerBlockSendDataBuf_;
  std::unique_ptr<transport::Buffer> smallerBlockRecvDataBuf_;

  std::vector<std::unique_ptr<transport::Buffer>> largerBlockSendDataBufs_;
  std::vector<std::unique_ptr<transport::Buffer>> largerBlockRecvDataBufs_;

  std::vector<size_t> sendCounts_;
  std::vector<size_t> recvCounts_;
  size_t sendCountToLargerBlock_;

  int dummy_;
  std::vector<std::unique_ptr<transport::Buffer>> sendNotificationBufs_;
  std::vector<std::unique_ptr<transport::Buffer>> recvNotificationBufs_;

  std::unique_ptr<LocalOp<T>> reduceBeforeFirstSend_;
  std::unique_ptr<LocalOp<T>> reduceBeforeFirstRecv_;

  std::unique_ptr<LocalOp<T> > localReduceOp_;
  std::unique_ptr<LocalOp<T> > localBroadcastOp_;

  // buffer where data is received prior to being reduced
  typename W::Pointer recvBuf_;

  typename W::Pointer scratchPtrForFirstSend_;
  typename W::Pointer scratchPtrForFirstRecv_;

  std::vector<CudaDevicePointer<T>> devicePtrsForFirstSend_;
  std::vector<CudaDevicePointer<T>> devicePtrsForFirstRecv_;

  std::vector<typename W::Pointer> scratchPtrForBroadcast_;
  std::vector<std::vector<CudaDevicePointer<T>>> devicePtrsForBroadcast_;
  std::vector<std::unique_ptr<LocalOp<T>>> broadcastOps_;

  bool pipelined_;

  // for non-power-of-two number of processes, partition the processes into
  // binary blocks and keep track of which block each process is in, as well as
  // the adjoining larger and smaller blocks (with which communication will be
  // required)
  uint32_t offsetToMyBinaryBlock_;
  uint32_t myBinaryBlockSize_;
  uint32_t stepsWithinBlock_;
  uint32_t rankInBinaryBlock_;
  uint32_t nextSmallerBlockSize_;
  uint32_t nextLargerBlockSize_;

  int slotOffset_;
};

} // namespace gloo
