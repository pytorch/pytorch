/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/ibverbs/address.h"

#include <string.h>

#include <array>

#include "gloo/common/logging.h"

namespace gloo {
namespace transport {
namespace ibverbs {

Address::Address() {
  memset(&addr_, 0, sizeof(addr_));
}

Address::Address(const std::vector<char>& bytes) {
  GLOO_ENFORCE_EQ(sizeof(addr_), bytes.size());
  memcpy(&addr_, bytes.data(), sizeof(addr_));
}

std::vector<char> Address::bytes() const {
  std::vector<char> bytes(sizeof(addr_));
  memcpy(bytes.data(), &addr_, sizeof(addr_));
  return bytes;
}

std::string Address::str() const {
  std::array<char, 128> buf;
  snprintf(
      buf.data(),
      buf.size(),
      "LID: %d QPN: %d PSN: %d",
      addr_.lid,
      addr_.qpn,
      addr_.psn);
  return std::string(buf.data());
}

} // namespace ibverbs
} // namespace transport
} // namespace gloo
