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
#include "gloo/context.h"

namespace gloo {

template <typename T>
class CudaAllreduceRing : public Allreduce<T> {
 public:
  CudaAllreduceRing(
    const std::shared_ptr<Context>& context,
    std::vector<T*> ptrs,
    int count);

  virtual ~CudaAllreduceRing();

  virtual void run() override;

 protected:
  // Forward declaration to prevent inclusion of CUDA headers.
  // Definition in cuda_allreduce_ring.cc.
  struct ptr;
  std::vector<struct ptr> ptrs_;

  int count_;
  int bytes_;

  std::unique_ptr<transport::Pair>& leftPair_;
  std::unique_ptr<transport::Pair>& rightPair_;

  T* inbox_;
  T* outbox_;
  std::unique_ptr<transport::Buffer> sendDataBuf_;
  std::unique_ptr<transport::Buffer> recvDataBuf_;

  int dummy_;
  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;
};

} // namespace gloo
