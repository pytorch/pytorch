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
class CudaBroadcastOneToAll : public Algorithm {
 public:
  CudaBroadcastOneToAll(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      int count,
      int rootRank = 0,
      int rootPointerRank = 0,
      const std::vector<cudaStream_t>& streams = std::vector<cudaStream_t>());

  virtual ~CudaBroadcastOneToAll() = default;

  virtual void run() override;

 protected:
  // Both workspace types have their own initialization function.
  template <typename U = W>
  void init(
      typename std::enable_if<std::is_same<U, CudaHostWorkspace<T> >::value,
                              typename U::Pointer>::type* = 0);

  std::vector<CudaDevicePointer<T> > devicePtrs_;
  std::vector<CudaStream> streams_;
  typename W::Pointer scratch_;

  const int count_;
  const int bytes_;
  const int rootRank_;
  const int rootPointerRank_;
  const bool synchronizeDeviceOutputs_;

  // For the sender (root)
  using forSender = struct {
    int dummy;
    std::unique_ptr<transport::Buffer> clearToSendBuffer;
    std::unique_ptr<transport::Buffer> sendBuffer;
  };

  std::vector<std::unique_ptr<forSender>> sender_;

  // For all receivers
  using forReceiver = struct {
    int dummy;
    std::unique_ptr<transport::Buffer> clearToSendBuffer;
    std::unique_ptr<transport::Buffer> recvBuffer;
  };

  std::unique_ptr<forReceiver> receiver_;

  // For local broadcast
  std::unique_ptr<LocalOp<T> > localBroadcastOp_;
};

} // namespace gloo
