/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "options.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sstream>

namespace gloo {
namespace benchmark {

static void usage(int status, const char* argv0) {
  if (status != EXIT_SUCCESS) {
    fprintf(stderr, "Try `%s --help' for more information.\n", argv0);
    exit(status);
  }

  fprintf(stderr, "Usage: %s [OPTIONS] BENCHMARK\n", argv0);

#define X(x) fputs(x "\n", stderr);
  X("");
  X("Participation:");
  X("  -s, --size=SIZE        Number of processes");
  X("  -r, --rank=RANK        Rank of this process");
  X("");
  X("Rendezvous:");
  X("  -h, --redis-host=HOST  Host name of Redis server");
  X("  -p, --redis-port=PORT  Port number of Redis server");
  X("  -x, --prefix=PREFIX    Rendezvous prefix (unique for this run)");
  X("");
  X("Transport:");
  X("  -t, --transport=TRANSPORT Transport to use (tcp, ibverbs, ...)");
  X("      --sync=BOOL           Switch pairs to sync mode (default: false)");
  X("      --busy-poll=BOOL      Busy-poll in sync mode (default: false)");
  X("");
  X("Benchmark parameters:");
  X("      --verify           Verify result first iteration (if applicable)");
  X("      --inputs           Number of input buffers");
  X("      --elements         Number of floats to use per input buffer");
  X("      --iteration-count  Number of iterations to run benchmark for");
  X("      --iteration-time   Time to run benchmark for (default: 2s)");
  X("      --nanos            Display timing data in nanos instead of micros");
  X("      --gpudirect        Use GPUDirect (CUDA only)");
  X("      --halfprecision    Use 16-bit floating point values");
  X("      --destinations     Number of separate destinations per host in "
                              "pairwise exchange benchmark");
  X("");
  X("BENCHMARK is one of:");
  X("  allreduce_ring");
  X("  allreduce_ring_chunked");
  X("  allreduce_halving_doubling");
  X("  barrier_all_to_all");
  X("  broadcast_one_to_all");
  X("  pairwise_exchange");
  X("");

  exit(status);
}

static long argToNanos(char** argv, const char* arg) {
  std::stringstream ss(arg);
  long num = 1;
  std::string unit = "s";
  ss >> num >> unit;
  if (unit == "s") {
    return num * 1000 * 1000 * 1000;
  } else if (unit == "ms") {
    return num * 1000 * 1000;
  } else {
    fprintf(stderr, "%s: invalid duration: %s\n", argv[0], arg);
    usage(EXIT_FAILURE, argv[0]);
  }

  return -1;
}

struct options parseOptions(int argc, char** argv) {
  struct options result;

  static struct option long_options[] = {
      {"rank", required_argument, nullptr, 'r'},
      {"size", required_argument, nullptr, 's'},
      {"redis-host", required_argument, nullptr, 'h'},
      {"redis-port", required_argument, nullptr, 'p'},
      {"prefix", required_argument, nullptr, 'x'},
      {"transport", required_argument, nullptr, 't'},
      {"verify", no_argument, nullptr, 0x1001},
      {"elements", required_argument, nullptr, 0x1002},
      {"iteration-count", required_argument, nullptr, 0x1003},
      {"iteration-time", required_argument, nullptr, 0x1004},
      {"sync", required_argument, nullptr, 0x1005},
      {"nanos", no_argument, nullptr, 0x1006},
      {"busy-poll", required_argument, nullptr, 0x1007},
      {"inputs", required_argument, nullptr, 0x1008},
      {"gpudirect", no_argument, nullptr, 0x1009},
      {"halfprecision", no_argument, nullptr, 0x100a},
      {"destinations", required_argument, nullptr, 0x100b},
      {"help", no_argument, nullptr, 0xffff},
      {nullptr, 0, nullptr, 0}};

  int opt;
  while (1) {
    int option_index = 0;
    opt = getopt_long(argc, argv, "r:s:h:p:x:t:", long_options, &option_index);
    if (opt == -1) {
      break;
    }

    switch (opt) {
      case 'r': {
        result.contextRank = atoi(optarg);
        break;
      }
      case 's': {
        result.contextSize = atoi(optarg);
        break;
      }
      case 'h': {
        result.redisHost = std::string(optarg, strlen(optarg));
        break;
      }
      case 'p': {
        result.redisPort = atoi(optarg);
        break;
      }
      case 'x': {
        result.prefix = std::string(optarg, strlen(optarg));
        break;
      }
      case 't': {
        result.transport = std::string(optarg, strlen(optarg));
        break;
      }
      case 0x1001: // --verify
      {
        result.verify = true;
        break;
      }
      case 0x1002: // --elements
      {
        result.elements = atoi(optarg);
        break;
      }
      case 0x1003: // --iteration-count
      {
        result.iterationCount = atoi(optarg);
        break;
      }
      case 0x1004: // --iteration-time
      {
        result.iterationTimeNanos = argToNanos(argv, optarg);
        break;
      }
      case 0x1005: // --sync
      {
        result.sync =
          atoi(optarg) == 1 ||
          tolower(optarg[0])== 't' ||
          tolower(optarg[0])== 'y';
        break;
      }
      case 0x1006: // --nanos
      {
        result.showNanos = true;
        break;
      }
      case 0x1007: // --busy-poll
      {
        result.busyPoll =
          atoi(optarg) == 1 ||
          tolower(optarg[0])== 't' ||
          tolower(optarg[0])== 'y';
        break;
      }
      case 0x1008: // --inputs
      {
        result.inputs = atoi(optarg);
        break;
      }
      case 0x1009: // --gpudirect
      {
        result.gpuDirect = true;
        break;
      }
      case 0x100a: // --halfprecision
      {
        result.halfPrecision = true;
        break;
      }
      case 0x100b: // --destinations
      {
        result.destinations = atoi(optarg);
        break;
      }
      case 0xffff: // --help
      {
        usage(EXIT_SUCCESS, argv[0]);
        break;
      }
      default: {
        usage(EXIT_FAILURE, argv[0]);
        break;
      }
    }
  }

#ifdef GLOO_USE_MPI
  // Use MPI if started through mpirun
  result.mpi = (getenv("OMPI_UNIVERSE_SIZE") != nullptr);
#endif

  if (result.busyPoll && !result.sync) {
    fprintf(stderr, "%s: busy poll can only be used with sync mode\n", argv[0]);
    usage(EXIT_FAILURE, argv[0]);
  }

  if (optind != (argc - 1)) {
    fprintf(stderr, "%s: missing benchmark specifier\n", argv[0]);
    usage(EXIT_FAILURE, argv[0]);
  }

  result.benchmark = argv[optind];
  return result;
}

} // namespace benchmark
} // namespace gloo
