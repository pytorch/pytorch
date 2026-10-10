/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace libkineto {

struct ITraceActivity;

class ActivityTraceInterface {
 public:
  virtual ~ActivityTraceInterface() = default;
  virtual const std::vector<const ITraceActivity*>* activities() {
    return nullptr;
  }
  virtual void save([[maybe_unused]] const std::string& path) {}
};

} // namespace libkineto
