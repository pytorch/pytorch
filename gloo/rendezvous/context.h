/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/context.h"
#include "gloo/rendezvous/store.h"
#include "gloo/transport/device.h"

namespace gloo {
namespace rendezvous {

class Context : public ::gloo::Context {
 public:
  Context(int rank, int size);
  virtual ~Context();

  void connectFullMesh(
      Store& store,
      std::shared_ptr<transport::Device>& dev);
};

} // namespace rendezvous

} // namespace gloo
