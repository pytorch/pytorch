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
#include <vector>

#include "gloo/common/error.h"
#include "gloo/context.h"
#include "gloo/rendezvous/store.h"
#include "gloo/transport/device.h"

namespace gloo {
namespace rendezvous {

class ContextFactory;

class Context : public ::gloo::Context {
  friend class ContextFactory;
 public:
  Context(int rank, int size);
  virtual ~Context();

  void connectFullMesh(
      Store& store,
      std::shared_ptr<transport::Device>& dev);
 protected:
  void setPairs(std::vector<std::unique_ptr<transport::Pair>>&& pairs);
  std::vector<char> extractAddress(std::vector<char>& allAddrs, int i);
};

class ContextFactory {
 public:
  // Assume a pair's address is no bigger than 128 bytes
  static constexpr auto kMaxAddressSize = 128;

  explicit ContextFactory(std::shared_ptr<::gloo::Context> backingContext);

  std::shared_ptr<::gloo::Context> makeContext(
    std::shared_ptr<transport::Device>& dev);

 protected:
  std::shared_ptr<::gloo::Context> backingContext_;

  std::vector<std::vector<char>> recvData_;
  std::vector<std::vector<char>> sendData_;

  std::vector<std::unique_ptr<transport::Buffer>> recvBuffers_;
  std::vector<std::unique_ptr<transport::Buffer>> sendBuffers_;

  std::vector<int> recvNotificationData_;
  std::vector<std::unique_ptr<transport::Buffer>> recvNotificationBuffers_;

  std::vector<int> sendNotificationData_;
  std::vector<std::unique_ptr<transport::Buffer>> sendNotificationBuffers_;
};


} // namespace rendezvous

} // namespace gloo
