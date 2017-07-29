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
#include "gloo/transport/device.h"

#include <mpi.h>

namespace gloo {
namespace mpi {

class Context : public ::gloo::Context {
 public:
  explicit Context(const MPI_Comm& comm);
  virtual ~Context();

  void connectFullMesh(std::shared_ptr<transport::Device>& dev);

 protected:
  MPI_Comm comm_;
};

} // namespace mpi
} // namespace gloo
