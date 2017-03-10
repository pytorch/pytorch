/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <cassert>
#include <iostream>

#include "gloo/mpi/context.h"
#include "gloo/transport/tcp/device.h"
#include "gloo/allreduce_ring.h"

int main(int argc, char** argv) {
  int rv;

  rv = MPI_Init(&argc, &argv);
  assert(rv == MPI_SUCCESS);

  // We'll use the TCP transport in this example
  auto dev = gloo::transport::tcp::CreateDevice("localhost");

  // Create Gloo context from MPI communicator
  auto context = std::make_shared<gloo::mpi::Context>(MPI_COMM_WORLD);
  context->connectFullMesh(dev);

  // Create and run simple allreduce
  int rank = context->rank;
  gloo::AllreduceRing<int> allreduce(context, {&rank}, 1);
  allreduce.run();
  std::cout << "Result: " << rank << std::endl;

  rv = MPI_Finalize();
  assert(rv == MPI_SUCCESS);
  return 0;
}
