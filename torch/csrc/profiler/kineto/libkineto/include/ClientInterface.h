/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
namespace libkineto {

class ClientInterface {
 public:
  virtual ~ClientInterface() = default;
  virtual void init() = 0;
  virtual void prepare(bool, bool, bool, bool, bool) = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual void start_memory_profile() = 0;
  virtual void stop_memory_profile() = 0;
  virtual void export_memory_profile(const std::string&) = 0;
};

} // namespace libkineto
