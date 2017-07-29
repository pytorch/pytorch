/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/context.h"

#include "gloo/common/logging.h"

namespace gloo {

Context::Context(int rank, int size)
    : rank(rank),
      size(size),
      slot_(0) {
  GLOO_ENFORCE_GE(rank, 0);
  GLOO_ENFORCE_LT(rank, size);
  GLOO_ENFORCE_GE(size, 2);
}

Context::~Context() {
}

std::unique_ptr<transport::Pair>& Context::getPair(int i) {
  return pairs_.at(i);
}

int Context::nextSlot(int numToSkip) {
  GLOO_ENFORCE_GT(numToSkip, 0);
  auto temp = slot_;
  slot_ += numToSkip;
  return temp;
}

} // namespace gloo
