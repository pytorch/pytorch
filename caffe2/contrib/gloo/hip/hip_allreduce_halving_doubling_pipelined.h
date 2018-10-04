/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

namespace gloo {

template <typename T, typename W = HipHostWorkspace<T>>
class HipAllreduceHalvingDoublingPipelined
    : public HipAllreduceHalvingDoubling<T, W> {
 public:
  HipAllreduceHalvingDoublingPipelined(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const std::vector<hipStream_t>& streams = std::vector<hipStream_t>())
      : HipAllreduceHalvingDoubling<T, W>(
            context,
            ptrs,
            count,
            streams,
            true) {}
};

} // namespace gloo
