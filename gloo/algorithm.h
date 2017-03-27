/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <memory>

#include "gloo/common.h"
#include "gloo/context.h"
#include "gloo/math.h"

namespace gloo {

class Algorithm {
 public:
  explicit Algorithm(const std::shared_ptr<Context>&);
  virtual ~Algorithm() = 0;

  virtual void run() = 0;

 protected:
  std::shared_ptr<Context> context_;

  const int contextRank_;
  const int contextSize_;

  std::unique_ptr<transport::Pair>& getPair(int i);

  // Helpers for ring algorithms
  std::unique_ptr<transport::Pair>& getLeftPair();
  std::unique_ptr<transport::Pair>& getRightPair();
};

} // namespace gloo
