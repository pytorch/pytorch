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

void Context::setPairs(std::vector<std::unique_ptr<transport::Pair>>&& pairs) {
  pairs_ = std::move(pairs);
}


ContextFactory::ContextFactory(std::shared_ptr<::gloo::Context> backingContext)
    : backingContext_(backingContext) {

  // We make sure that we have a fully connected context
  for(int i = 0; i < backingContext_->size; ++i) {
    if (i == backingContext_->rank) {
      continue;
    }
    try {
      GLOO_ENFORCE(
        backingContext_->getPair(i) != nullptr,
        "Missing pair in backing context");
    } catch(std::out_of_range& e) {
      GLOO_THROW("Backing context not fully connected");
    }
  }
}

std::shared_ptr<::gloo::Context> ContextFactory::makeContext(
  std::shared_ptr<transport::Device>& dev) {

  using syncData = struct {
    std::vector<char> me;
  };

  auto toReturn = std::make_shared<Context>(backingContext_->rank,
                                            backingContext_->size);
  std::vector<std::unique_ptr<transport::Pair>> pairs(toReturn->size);

  size_t maxAddressSize = 0;
  int startingSlot = backingContext_->nextSlot(
    backingContext_->size*backingContext_->size);

  std::vector<syncData> pairInfo(toReturn->size);

  for (int i = 0; i < toReturn->size; ++i) {
    if (i == toReturn->rank) {
      continue;
    }
    auto& tPair = backingContext_->getPair(i);
    pairs[i] = dev->createPair();
    pairInfo[i].me = pairs[i]->address().bytes();
    tPair->createSendBuffer(startingSlot +
                            backingContext_->rank*backingContext_->size+i,
                            pairInfo[i].me.data(),
                            pairInfo[i].me.size())->send();
    maxAddressSize = std::max(maxAddressSize, pairInfo[i].me.size());
  }

  std::vector<char> recvBuffer(maxAddressSize);
  for (int i = 0; i < toReturn->size; ++i) {
    if (i == toReturn->rank) {
      continue;
    }
    auto& tPair = backingContext_->getPair(i);
    tPair->createRecvBuffer(startingSlot +
                            i*backingContext_->size+backingContext_->rank,
                            recvBuffer.data(), recvBuffer.size())->waitRecv();
    pairs[i]->connect(recvBuffer);
  }
  toReturn->setPairs(std::move(pairs));
  return std::static_pointer_cast<::gloo::Context>(toReturn);
}

} // namespace rendezvous
} // namespace gloo
