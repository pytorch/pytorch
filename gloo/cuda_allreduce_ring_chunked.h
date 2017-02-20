/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/allreduce.h"
#include "gloo/cuda.h"

namespace gloo {

template <typename T>
class CudaAllreduceRingChunked : public Allreduce<T> {
 public:
  CudaAllreduceRingChunked(
    const std::shared_ptr<Context>& context,
    const std::vector<T*>& ptrs,
    int count,
    const std::vector<cudaStream_t>& streams = std::vector<cudaStream_t>());

  virtual ~CudaAllreduceRingChunked();

  virtual void run() override;

 protected:
  void copyChunkAtOffset(int chunkOffset);

  std::vector<CudaDevicePointer<T> > devicePtrs_;
  std::vector<T*> hostPtrs_;

  const int count_;
  const int bytes_;

  std::unique_ptr<transport::Pair>& leftPair_;
  std::unique_ptr<transport::Pair>& rightPair_;

  size_t chunks_;
  size_t chunkSize_;
  size_t chunkBytes_;

  std::array<T*, 2> inbox_;
  std::array<std::unique_ptr<transport::Buffer>, 2> sendDataBuf_;
  std::array<std::unique_ptr<transport::Buffer>, 2> recvDataBuf_;

  int dummy_;
  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;
};

} // namespace gloo
