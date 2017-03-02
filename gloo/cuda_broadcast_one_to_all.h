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

namespace gloo {

template <typename T>
class CudaBroadcastOneToAll : public Broadcast<T> {
 public:
  CudaBroadcastOneToAll(
      const std::shared_ptr<Context>& context,
      T* ptr,
      int count,
      int rootRank = 0,
      cudaStream_t stream = kStreamNotSet);

  virtual ~CudaBroadcastOneToAll();

  virtual void run() override;

 protected:
  CudaDevicePointer<T> devicePtr_;
  T* hostPtr_;

  const int count_;
  const int bytes_;
  const bool synchronizeDeviceOutputs_;

  // For the sender (root)
  std::vector<std::unique_ptr<transport::Buffer>> sendDataBuffers_;

  // For all receivers
  std::unique_ptr<transport::Buffer> recvDataBuffer_;
};

} // namespace gloo
