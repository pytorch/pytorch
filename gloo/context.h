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

#include "gloo/transport/pair.h"

namespace gloo {

class Context {
 public:
  Context(int rank, int size);
  virtual ~Context();

  const int rank;
  const int size;

  std::unique_ptr<transport::Pair>& getPair(int i);

  int nextSlot(int numToSkip = 1);

 protected:
  std::vector<std::unique_ptr<transport::Pair>> pairs_;
  int slot_;
};

} // namespace gloo
