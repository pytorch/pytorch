/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <functional>
#include <thread>
#include <vector>

#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// Test fixture.
class BufferTest : public BaseTest {};

TEST_F(BufferTest, RemoteOffset) {
  constexpr auto processCount = 2;

  spawn(processCount, [&](std::shared_ptr<Context> context) {
      std::array<float, processCount> data;
      std::unique_ptr<transport::Buffer> sendBuffer;
      std::unique_ptr<transport::Buffer> recvBuffer;

      if (context->rank == 0) {
        auto& other = context->getPair(1);
        sendBuffer = other->createSendBuffer(
          0, data.data(), data.size() * sizeof(float));
        recvBuffer = other->createRecvBuffer(
          1, data.data(), data.size() * sizeof(float));
      }
      if (context->rank == 1) {
        auto& other = context->getPair(0);
        recvBuffer = other->createRecvBuffer(
          0, data.data(), data.size() * sizeof(float));
        sendBuffer = other->createSendBuffer(
          1, data.data(), data.size() * sizeof(float));
      }

      // Set value indexed on this process' rank
      data[context->rank] = context->rank + 1000;

      // Send value to the remote buffer (using offset)
      auto offset = context->rank * sizeof(float);
      sendBuffer->send(offset, sizeof(float), offset);
      sendBuffer->waitSend();

      // Wait for receive
      recvBuffer->waitRecv();

      // Both processes have written to each other's buffer
      // at an offset equal to their rank. Their buffers now
      // contain the same values.
      for (auto i = 0; i < data.size(); i++) {
        ASSERT_EQ(1000 + i, data[i]);
      }
    });
}

} // namespace
} // namespace test
} // namespace gloo
