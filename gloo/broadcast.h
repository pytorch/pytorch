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
#include "gloo/common/logging.h"

namespace gloo {

template <typename T>
class Broadcast : public Algorithm {
 public:
  Broadcast(
      const std::shared_ptr<Context>& context,
      int rootRank,
      int rootPointerRank)
      : Algorithm(context),
        rootRank_(rootRank),
        rootPointerRank_(rootPointerRank) {
    GLOO_ENFORCE_GE(rootRank_, 0);
    GLOO_ENFORCE_LT(rootRank_, contextSize_);
  }

  int getRootRank() const {
    return rootRank_;
  }

  int getRootPointerRank() const {
    return rootPointerRank_;
  }

  virtual ~Broadcast(){};

 protected:
  const int rootRank_;
  const int rootPointerRank_;
};

} // namespace gloo
