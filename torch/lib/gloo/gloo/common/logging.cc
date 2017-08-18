/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/common/logging.h"

#include <algorithm>
#include <numeric>

namespace gloo {

EnforceNotMet::EnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg)
    : msg_stack_{MakeString(
          "[enforce fail at ",
          file,
          ":",
          line,
          "] ",
          condition,
          ". ",
          msg)} {
  full_msg_ = this->msg();
}

std::string EnforceNotMet::msg() const {
  return std::accumulate(msg_stack_.begin(), msg_stack_.end(), std::string(""));
}

const char* EnforceNotMet::what() const noexcept {
  return full_msg_.c_str();
}

} // namespace gloo
