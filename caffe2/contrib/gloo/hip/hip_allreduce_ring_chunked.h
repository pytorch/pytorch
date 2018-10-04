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
#include "hip.h"
#include "hip_workspace.h"

namespace gloo {

template <typename T, typename W = HipHostWorkspace<T> >
class HipAllreduceRingChunked : public Algorithm {
 public:
  HipAllreduceRingChunked(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const std::vector<hipStream_t>& streams = std::vector<hipStream_t>());

  virtual ~HipAllreduceRingChunked();

  virtual void run() override;

 protected:
  int getChunkOffset(int round);
  void copyChunkAtOffset(int chunkOffset);

  // Both workspace types have their own initialization function.
  template <typename U = W>
  void init(
    typename std::enable_if<std::is_same<U, HipHostWorkspace<T> >::value,
    typename U::Pointer>::type* = 0);

  template <typename U = W>
  void init(
    typename std::enable_if<std::is_same<U, HipDeviceWorkspace<T> >::value,
    typename U::Pointer>::type* = 0);

  std::vector<HipDevicePointer<T> > devicePtrs_;
  std::vector<HipStream> streams_;
  typename W::Pointer scratch_;
  HipStream* scratchStream_;

  const int count_;
  const int bytes_;
  const bool synchronizeDeviceOutputs_;
  const HipReductionFunction<T>* fn_;

  size_t chunks_;
  size_t chunkSize_;
  size_t chunkBytes_;

  struct ChunkContext;
  std::vector<ChunkContext> chunkContext_;

  std::array<typename W::Pointer, 2> inbox_;
  std::array<std::unique_ptr<transport::Buffer>, 2> sendDataBuf_;
  std::array<std::unique_ptr<transport::Buffer>, 2> recvDataBuf_;

  int dummy_;
  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;
};

} // namespace gloo
