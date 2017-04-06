/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/algorithm.h"
#include "gloo/cuda.h"
#include "gloo/cuda_workspace.h"

namespace gloo {

template <typename T, typename W = CudaHostWorkspace<T> >
class CudaAllreduceRing : public Algorithm {
 public:
  CudaAllreduceRing(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const std::vector<cudaStream_t>& streams = std::vector<cudaStream_t>());

  virtual ~CudaAllreduceRing() = default;

  virtual void run() override;

 protected:
  // Both workspace types have their own initialization function.
  template <typename U = W>
  void init(
      typename std::enable_if<std::is_same<U, CudaHostWorkspace<T> >::value,
                              typename U::Pointer>::type* = 0);

  template <typename U = W>
  void init(
      typename std::enable_if<std::is_same<U, CudaDeviceWorkspace<T> >::value,
                              typename U::Pointer>::type* = 0);

  std::vector<CudaDevicePointer<T> > devicePtrs_;
  std::vector<CudaStream> streams_;
  typename W::Pointer scratch_;

  const int count_;
  const int bytes_;
  const bool synchronizeDeviceOutputs_;
  const CudaReductionFunction<T>* fn_;

  std::unique_ptr<transport::Pair>& leftPair_;
  std::unique_ptr<transport::Pair>& rightPair_;

  std::unique_ptr<LocalOp<T> > localReduceOp_;
  std::unique_ptr<LocalOp<T> > localBroadcastOp_;

  typename W::Pointer inbox_;
  typename W::Pointer outbox_;
  std::unique_ptr<transport::Buffer> sendDataBuf_;
  std::unique_ptr<transport::Buffer> recvDataBuf_;

  int dummy_;
  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;
};

} // namespace gloo
