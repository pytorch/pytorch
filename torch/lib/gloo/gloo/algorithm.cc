/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/algorithm.h"

#include "gloo/common/logging.h"

namespace gloo {

// Do host reduce/bcast on buf size less than 256KB and device reduce above
const size_t kOnDeviceThreshold = 256 * 1024;

Algorithm::Algorithm(const std::shared_ptr<Context>& context)
    : context_(context),
      contextRank_(context_->rank),
      contextSize_(context_->size) {}

// Have to provide implementation for pure virtual destructor.
Algorithm::~Algorithm() {}

std::unique_ptr<transport::Pair>& Algorithm::getPair(int i) {
  return context_->getPair(i);
}

// Helper for ring algorithms
std::unique_ptr<transport::Pair>& Algorithm::getLeftPair() {
  auto rank = (context_->size + context_->rank - 1) % context_->size;
  GLOO_ENFORCE(context_->getPair(rank), "pair missing (index ", rank, ")");
  return context_->getPair(rank);
}

// Helper for ring algorithms
std::unique_ptr<transport::Pair>& Algorithm::getRightPair() {
  auto rank = (context_->rank + 1) % context_->size;
  GLOO_ENFORCE(context_->getPair(rank), "pair missing (index ", rank, ")");
  return context_->getPair(rank);
}

} // namespace gloo
