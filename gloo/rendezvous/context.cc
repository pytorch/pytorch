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
    store.wait({key.str()}, dev->getTimeout());

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
  for (auto i = 0; i < backingContext_->size; i++) {
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

  auto slot = backingContext_->nextSlot();
  auto notificationSlot = backingContext_->nextSlot();

  // Create buffers we'll later use to communicate pair addresses
  recvData_.resize(backingContext_->size);
  sendData_.resize(backingContext_->size);
  recvBuffers_.resize(backingContext_->size);
  sendBuffers_.resize(backingContext_->size);
  recvNotificationData_.resize(backingContext_->size);
  sendNotificationData_.resize(backingContext_->size);
  recvNotificationBuffers_.resize(backingContext_->size);
  sendNotificationBuffers_.resize(backingContext_->size);
  for (auto i = 0; i < backingContext_->size; i++) {
    if (i == backingContext_->rank) {
      continue;
    }

    // Allocate memory for recv/send
    recvData_[i].resize(kMaxAddressSize);
    sendData_[i].resize(kMaxAddressSize);

    // Create pair
    auto& pair = backingContext_->getPair(i);

    // Create payload buffers
    {
      auto recvPtr = recvData_[i].data();
      auto recvSize = recvData_[i].size();
      recvBuffers_[i] = pair->createRecvBuffer(slot, recvPtr, recvSize);
      auto sendPtr = sendData_[i].data();
      auto sendSize = sendData_[i].size();
      sendBuffers_[i] = pair->createSendBuffer(slot, sendPtr, sendSize);
    }

    // Create notification buffers
    {
      auto recvPtr = &recvNotificationData_[i];
      auto recvSize = sizeof(*recvPtr);
      recvNotificationBuffers_[i] =
        pair->createRecvBuffer(notificationSlot, recvPtr, recvSize);
      auto sendPtr = &sendNotificationData_[i];
      auto sendSize = sizeof(*sendPtr);
      sendNotificationBuffers_[i] =
        pair->createSendBuffer(notificationSlot, sendPtr, sendSize);
    }
  }
}

std::shared_ptr<::gloo::Context> ContextFactory::makeContext(
    std::shared_ptr<transport::Device>& dev) {
  auto context = std::make_shared<Context>(
      backingContext_->rank,
      backingContext_->size);
  std::vector<std::unique_ptr<transport::Pair>> pairs(context->size);

  // Assume it's the same for all pairs on a device
  size_t addressSize = 0;

  // Create pairs
  for (auto i = 0; i < context->size; i++) {
    if (i == context->rank) {
      continue;
    }

    auto pair = dev->createPair();
    auto address = pair->address().bytes();
    addressSize = address.size();
    pairs[i] = std::move(pair);

    // Send address of new pair to peer
    GLOO_ENFORCE_LE(addressSize, sendData_[i].size());
    sendData_[i].assign(address.begin(), address.end());
    sendBuffers_[i]->send(0, addressSize);
  }

  // Wait for remote addresses and connect peers
  for (auto i = 0; i < context->size; i++) {
    if (i == context->rank) {
      continue;
    }

    recvBuffers_[i]->waitRecv();
    auto& data = recvData_[i];
    auto address = std::vector<char>(data.begin(), data.begin() + addressSize);
    pairs[i]->connect(address);

    // Notify peer that we've consumed the payload
    sendNotificationBuffers_[i]->send();
  }

  // Wait for notification from peers
  for (auto i = 0; i < context->size; i++) {
    if (i == context->rank) {
      continue;
    }

    recvNotificationBuffers_[i]->waitRecv();
  }

  context->setPairs(std::move(pairs));
  return std::static_pointer_cast<::gloo::Context>(context);
}

} // namespace rendezvous
} // namespace gloo
