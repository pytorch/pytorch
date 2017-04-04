/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "runner.h"

#include <iomanip>
#include <iostream>

#include "gloo/barrier_all_to_one.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/common/logging.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/rendezvous/redis_store.h"
#include "gloo/transport/device.h"

#ifdef GLOO_USE_MPI
#include "gloo/mpi/context.h"
#endif

#ifdef BENCHMARK_TCP
#include "gloo/transport/tcp/device.h"
#endif

#ifdef BENCHMARK_IBVERBS
#include "gloo/transport/ibverbs/device.h"
#endif

namespace gloo {
namespace benchmark {

Runner::Runner(const options& options) : options_(options) {
#ifdef BENCHMARK_TCP
  if (options_.transport == "tcp") {
    transport::tcp::attr attr;
    device_ = transport::tcp::CreateDevice(attr);
  }
#endif
#ifdef BENCHMARK_IBVERBS
  if (options_.transport == "ibverbs") {
    transport::ibverbs::attr attr = {
      .name = options_.ibverbsDevice,
      .port = options_.ibverbsPort,
      .index = options_.ibverbsIndex,
    };
    device_ = transport::ibverbs::CreateDevice(attr);
  }
#endif
  GLOO_ENFORCE(device_, "Unknown transport: ", options_.transport);

#ifdef GLOO_USE_MPI
  if (options_.mpi) {
    auto rv = MPI_Init(nullptr, nullptr);
    GLOO_ENFORCE_EQ(rv, MPI_SUCCESS);
    MPI_Comm_rank(MPI_COMM_WORLD, &options_.contextRank);
    MPI_Comm_size(MPI_COMM_WORLD, &options_.contextSize);
  }
#endif

  // Create broadcast algorithm to synchronize between participants
  broadcast_.reset(
    new BroadcastOneToAll<long>(newContext(), {&broadcastValue_}, 1));

  // Create barrier for run-to-run synchronization
  barrier_.reset(new BarrierAllToOne(newContext()));
}

Runner::~Runner() {
#ifdef GLOO_USE_MPI
  if (options_.mpi) {
    MPI_Finalize();
  }
#endif
}

long Runner::broadcast(long value) {
  // Set value to broadcast only on root.
  // Otherwise it can race with the actual broadcast
  // operation writing to the same memory location.
  if (options_.contextRank == 0) {
    broadcastValue_ = value;
  }
  broadcast_->run();
  return broadcastValue_;
}

std::shared_ptr<Context> Runner::newContext() {
#ifdef GLOO_USE_MPI
  if (options_.mpi) {
    auto context = std::make_shared<::gloo::mpi::Context>(MPI_COMM_WORLD);
    context->connectFullMesh(device_);
    return context;
  }
#endif

  std::stringstream prefix;
  prefix << options_.prefix << "-" << prefixCounter_++;

  auto context =
    std::make_shared<rendezvous::Context>(
      options_.contextRank, options_.contextSize);

  rendezvous::RedisStore redisStore(options_.redisHost, options_.redisPort);
  rendezvous::PrefixStore prefixStore(prefix.str(), redisStore);
  context->connectFullMesh(prefixStore, device_);
  return context;
}

void Runner::run(BenchmarkFn& fn) {
  printHeader();

  if (options_.elements > 0) {
    run(fn, options_.elements);
    return;
  }

  // Run sweep over number of elements
  for (int i = 100; i <= 1000000; i *= 10) {
    std::vector<int> js = {i * 1, i * 2, i * 5};
    for (auto& j : js) {
      run(fn, j);
    }
  }
}

void Runner::run(BenchmarkFn& fn, int n) {
    auto context = newContext();
    auto benchmark = fn(context);
    benchmark->initialize(n);

    // Switch pairs to sync mode if configured to do so
    if (options_.sync) {
      for (int i = 0; i < context->size; i++) {
        auto& pair = context->getPair(i);
        if (pair) {
          pair->setSync(true, options_.busyPoll);
        }
      }
    }

    // Verify correctness of initial run
    if (options_.verify) {
      benchmark->run();
      benchmark->verify();
      barrier_->run();
    }

    // Switch mode based on iteration count or time spent
    auto iterations = options_.iterationCount;
    if (iterations <= 0) {
      GLOO_ENFORCE_GT(options_.iterationTimeNanos, 0);

      Distribution warmup;
      for (int i = 0; i < options_.warmupIterationCount; i++) {
        Timer dt;
        benchmark->run();
        warmup.add(dt);
      }

      // Broadcast duration of fastest iteration during warmup,
      // so all nodes agree on the number of iterations to run for.
      auto nanos = broadcast(warmup.min());
      iterations = options_.iterationTimeNanos / nanos;
    }

    // Main benchmark loop
    samples_.clear();
    for (int i = 0; i < iterations; i++) {
      Timer dt;
      benchmark->run();
      samples_.add(dt);
    }

    printDistribution(n);

    // Barrier to make sure everybody arrived here and the temporary
    // context and benchmark can be destructed.
    barrier_->run();
}

void Runner::printHeader() {
  if (options_.contextRank != 0) {
    return;
  }

  std::string suffix = "(us)";
  if (options_.showNanos) {
    suffix = "(ns)";
  }

  std::cout << std::setw(11) << "elements";
  std::cout << std::setw(11) << ("min " + suffix);
  std::cout << std::setw(11) << ("p50 " + suffix);
  std::cout << std::setw(11) << ("p99 " + suffix);
  std::cout << std::setw(11) << ("max " + suffix);
  std::cout << std::setw(11) << "samples";
  std::cout << std::endl;
}

void Runner::printDistribution(int elements) {
  if (options_.contextRank != 0) {
    return;
  }

  auto div = 1000;
  if (options_.showNanos) {
    div = 1;
  }

  std::cout << std::setw(11) << elements;
  std::cout << std::setw(11) << samples_.percentile(0.00) / div;
  std::cout << std::setw(11) << samples_.percentile(0.50) / div;
  std::cout << std::setw(11) << samples_.percentile(0.90) / div;
  std::cout << std::setw(11) << samples_.percentile(0.99) / div;
  std::cout << std::setw(11) << samples_.size();
  std::cout << std::endl;
}

} // namespace benchmark
} // namespace gloo
