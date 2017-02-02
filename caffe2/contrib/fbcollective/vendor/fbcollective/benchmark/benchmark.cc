#include <getopt.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <iostream>
#include <memory>

#include "fbcollective/allreduce_ring.h"
#include "fbcollective/allreduce_ring_chunked.h"
#include "fbcollective/barrier_all_to_all.h"
#include "fbcollective/broadcast_one_to_all.h"
#include "fbcollective/common/logging.h"
#include "fbcollective/context.h"
#include "fbcollective/rendezvous/prefix_store.h"
#include "fbcollective/rendezvous/redis_store.h"
#include "fbcollective/transport/device.h"

#ifdef BENCHMARK_TCP
#include "fbcollective/transport/tcp/device.h"
#endif

#ifdef BENCHMARK_IBVERBS
#include "fbcollective/transport/ibverbs/device.h"
#endif

using namespace fbcollective;

namespace {

class timer {
 public:
  timer() {
    start();
  }

  void start() {
    start_ = std::chrono::high_resolution_clock::now();
  }

  long ns() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::nanoseconds(now - start_).count();
  }

 protected:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

class Benchmark {
 public:
  explicit Benchmark(std::shared_ptr<Context>& context, int dataSize = 0)
      : context_(context), dataSize_(dataSize), algorithm_(nullptr) {
    if (dataSize_ > 0) {
      ptr_.reset(new float[dataSize_]);
      for (int i = 0; i < dataSize_; i++) {
        ptr_[i] = context->rank_;
      }
    }
  }

  virtual ~Benchmark() {
    if (algorithm_ != nullptr) {
      delete algorithm_;
    }
  }

  virtual void initialize() = 0;

  virtual void run() {
    algorithm_->Run();
  }

  virtual bool verify() = 0;

 protected:
  std::shared_ptr<Context> context_;
  const int dataSize_;
  std::unique_ptr<float[]> ptr_;
  Algorithm* algorithm_;
};

class AllreduceRingBenchmark : public Benchmark {
 public:
  explicit AllreduceRingBenchmark(std::shared_ptr<Context>& context, int nelem)
      : Benchmark(context, nelem) {}

  virtual void initialize() override {
    algorithm_ = new AllreduceRing<float>(context_, {ptr_.get()}, dataSize_);
  }

  virtual bool verify() override {
    auto expected = (context_->size_ * (context_->size_ - 1)) / 2;
    for (int i = 0; i < dataSize_; i++) {
      FBC_ENFORCE_EQ(expected, ptr_[i], "Mismatch at index ", i);
    }
    return true;
  }
};

class AllreduceRingChunkedBenchmark : public Benchmark {
 public:
  explicit AllreduceRingChunkedBenchmark(
      std::shared_ptr<Context>& context,
      int nelem)
      : Benchmark(context, nelem) {}

  virtual void initialize() override {
    algorithm_ =
        new AllreduceRingChunked<float>(context_, {ptr_.get()}, dataSize_);
  }

  virtual bool verify() override {
    auto expected = (context_->size_ * (context_->size_ - 1)) / 2;
    for (int i = 0; i < dataSize_; i++) {
      FBC_ENFORCE_EQ(expected, ptr_[i], "Mismatch at index ", i);
    }
    return true;
  }
};

class BarrierAllToAllBenchmark : public Benchmark {
 public:
  explicit BarrierAllToAllBenchmark(std::shared_ptr<Context>& context)
      : Benchmark(context) {}

  virtual void initialize() override {
    algorithm_ = new BarrierAllToAll(context_);
  }

  virtual bool verify() override {
    return true;
  }
};

class BroadcastOneToAllBenchmark : public Benchmark {
 public:
  explicit BroadcastOneToAllBenchmark(
      std::shared_ptr<Context>& context,
      int nelem)
      : Benchmark(context, nelem), rootRank_(0) {}

  virtual void initialize() override {
    algorithm_ = new BroadcastOneToAll<float>(
        context_, ptr_.get(), dataSize_, rootRank_);
  }

  virtual bool verify() override {
    for (int i = 0; i < dataSize_; i++) {
      FBC_ENFORCE_EQ(rootRank_, ptr_[i], "Mismatch at index ", i);
    }
    return true;
  }

 protected:
  const int rootRank_;
};

void usage(int /* unused */, char** argv) {
  fprintf(stderr, "Usage: %s [OPTIONS] BENCHMARK\n", argv[0]);
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -s SIZE   Number of participating processes\n");
  fprintf(stderr, "  -r RANK   Rank of this process\n");
  fprintf(stderr, "  -h HOST   Host name of Redis server (for rendezvous)\n");
  fprintf(stderr, "  -p PORT   Port number of Redis server (for rendezvous)\n");
  fprintf(stderr, "  -t TRANSPORT  Transport to use (tcp|...)\n");
  fprintf(stderr, "  -c        Verify result on first iteration\n");
  fprintf(stderr, "  -n NELEM  Number of floats\n");
  fprintf(stderr, "  -i NITER  Number of iterations\n");
  fprintf(stderr, "  -x PREFIX Rendezvous prefix (unique for this run)\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "BENCHMARK is one of:\n");
  fprintf(stderr, "  allreduce_ring\n");
  fprintf(stderr, "  allreduce_ring_chunked\n");
  fprintf(stderr, "  barrier_all_to_all\n");
  fprintf(stderr, "  broadcast_one_to_all\n");
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}

} // namespace

int main(int argc, char** argv) {
  int originalArgc = argc;
  char** originalArgv = argv;
  int contextSize = 0;
  int contextRank = 0;
  std::string redisHost;
  int redisPort = 6379;
  std::string transport = "tcp";
  bool verify = false;
  int nelem = 1000;
  int niters = 1000;
  std::string prefix = "prefix";

  int opt;
  while ((opt = getopt(argc, argv, "s:r:h:p:t:cn:i:x:")) != -1) {
    switch (opt) {
      case 's':
        contextSize = atoi(optarg);
        break;
      case 'r':
        contextRank = atoi(optarg);
        break;
      case 'h':
        redisHost = std::string(optarg, strlen(optarg));
        break;
      case 'f':
        redisPort = atoi(optarg);
        break;
      case 't':
        transport = std::string(optarg, strlen(optarg));
        break;
      case 'c':
        verify = true;
        break;
      case 'n':
        nelem = atoi(optarg);
        break;
      case 'i':
        niters = atoi(optarg);
        break;
      case 'x':
        prefix = std::string(optarg, strlen(optarg));
        break;
      default:
        usage(originalArgc, originalArgv);
        break;
    }
  }

  if (optind != (argc - 1)) {
    usage(originalArgc, originalArgv);
  }

  std::string algorithm(argv[optind]);
  std::function<std::shared_ptr<Benchmark>(std::shared_ptr<Context>&)> fn;
  if (algorithm == "allreduce_ring") {
    fn = [&](std::shared_ptr<Context>& context) {
      return std::make_shared<AllreduceRingBenchmark>(context, nelem);
    };
  } else if (algorithm == "allreduce_ring_chunked") {
    fn = [&](std::shared_ptr<Context>& context) {
      return std::make_shared<AllreduceRingChunkedBenchmark>(context, nelem);
    };
  } else if (algorithm == "barrier_all_to_all") {
    fn = [&](std::shared_ptr<Context>& context) {
      return std::make_shared<BarrierAllToAllBenchmark>(context);
    };
  } else if (algorithm == "broadcast_one_to_all") {
    fn = [&](std::shared_ptr<Context>& context) {
      return std::make_shared<BroadcastOneToAllBenchmark>(context, nelem);
    };
  }

  if (!fn) {
    FBC_ENFORCE(false, "Invalid algorithm: ", algorithm);
  }

  std::shared_ptr<transport::Device> device;
#ifdef BENCHMARK_TCP
  if (transport == "tcp") {
    transport::tcp::attr attr;
    device = transport::tcp::CreateDevice(attr);
  }
#endif
#ifdef BENCHMARK_IBVERBS
  if (transport == "ibverbs") {
    transport::ibverbs::attr attr = {
        .name = "mlx5_0", .port = 1, .index = 1,
    };
    device = transport::ibverbs::CreateDevice(attr);
  }
#endif
  FBC_ENFORCE(device, "Unknown transport: ", transport);

  auto context = std::make_shared<Context>(contextRank, contextSize);
  auto redisStore = std::unique_ptr<rendezvous::Store>(
      new rendezvous::RedisStore(redisHost, redisPort));
  auto prefixStore = std::unique_ptr<rendezvous::Store>(
      new rendezvous::PrefixStore(prefix, redisStore));
  context->connectFullMesh(*prefixStore, device);

  auto benchmark = fn(context);
  benchmark->initialize();

  // Verify correctness of initial run
  if (verify) {
    benchmark->run();
    FBC_ENFORCE(benchmark->verify());
  }

  {
    timer t;
    unsigned long runs = 0;
    unsigned long ns = 0;

    for (int i = 0; i < niters; i++) {
      timer dt;
      benchmark->run();
      auto rns = dt.ns();
      ns += rns;
      runs++;

      // Only write timing information on node 0
      if (context->rank_ == 0) {
        // Log current time every second to give some feedback
        if (t.ns() > (1000 * 1000 * 1000)) {
          std::cout << rns << "ns" << std::endl;
          t.start();
        }
      }
    }
  }

  return 0;
}
