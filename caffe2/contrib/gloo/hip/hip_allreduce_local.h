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

namespace gloo {

template <typename T>
class HipAllreduceLocal : public Algorithm {
 public:
  HipAllreduceLocal(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const std::vector<hipStream_t>& streams = std::vector<hipStream_t>());

  virtual ~HipAllreduceLocal() = default;

  virtual void run() override;

 protected:
  std::vector<HipDevicePointer<T> > devicePtrs_;
  std::vector<HipStream> streams_;
  const int count_;
  const int bytes_;
  const HipReductionFunction<T>* fn_;
  const bool synchronizeDeviceOutputs_;

  std::unique_ptr<LocalOp<T> > localReduceOp_;
  std::unique_ptr<LocalOp<T> > localBroadcastOp_;
};

} // namespace gloo
