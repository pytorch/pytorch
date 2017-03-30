/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/broadcast.h"
#include "gloo/cuda.h"
#include "gloo/cuda_collectives.h"

namespace gloo {

template <typename T>
class CudaBroadcastOneToAll : public Broadcast<T> {
 public:
  CudaBroadcastOneToAll(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      int count,
      int rootRank = 0,
      int rootPointerRank = 0,
      const std::vector<cudaStream_t>& streams = std::vector<cudaStream_t>());

  virtual ~CudaBroadcastOneToAll();

  virtual void run() override;

 protected:
  std::vector<CudaDevicePointer<T> > devicePtrs_;
  T* hostPtr_;

  const int count_;
  const int bytes_;
  const bool synchronizeDeviceOutputs_;

  // For the sender (root)
  std::vector<std::unique_ptr<transport::Buffer>> sendDataBuffers_;

  // For all receivers
  std::unique_ptr<transport::Buffer> recvDataBuffer_;

  // For local broadcast
  std::unique_ptr<LocalOp<T> > localBroadcastOp_;
};

} // namespace gloo
