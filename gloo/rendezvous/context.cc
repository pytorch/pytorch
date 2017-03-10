/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/rendezvous/context.h"

#include "gloo/common/logging.h"
#include "gloo/transport/address.h"

namespace gloo {
namespace rendezvous {

Context::Context(int rank, int size)
    : ::gloo::Context(rank, size) {
}

Context::~Context() {
}

void Context::connectFullMesh(
    rendezvous::Store& store,
    std::shared_ptr<transport::Device>& dev) {
  std::vector<std::unique_ptr<transport::Pair>> pairs(size);

  // Create pair to connect to every other node in the collective
  for (int i = 0; i < size; i++) {
    if (i == rank) {
      continue;
    }

    auto pair = dev->createPair();
    pairs[i] = std::move(pair);

    // Store address for pair for this rank
    std::ostringstream key;
    key << rank << "/" << i;
    store.set(key.str(), pairs[i]->address().bytes());
  }

  // Connect every pair
  for (int i = 0; i < size; i++) {
    if (i == rank) {
      continue;
    }

    // Wait for address of other side of this pair to become available
    std::ostringstream key;
    key << i << "/" << rank;
    store.wait({key.str()});

    // Connect to other side of this pair
    auto addr = store.get(key.str());
    pairs[i]->connect(addr);
  }

  pairs_ = std::move(pairs);
}

} // namespace rendezvous
} // namespace gloo
