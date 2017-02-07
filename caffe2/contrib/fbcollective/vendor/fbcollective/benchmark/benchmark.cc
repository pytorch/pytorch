#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>

#include <Eigen/Core>

#include "fbcollective/allreduce_ring.h"
#include "fbcollective/allreduce_ring_chunked.h"
#include "fbcollective/barrier_all_to_all.h"
#include "fbcollective/barrier_all_to_one.h"
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

#include "fbcollective/benchmark/options.h"
#include "fbcollective/benchmark/timer.h"

using namespace fbcollective;
using namespace fbcollective::benchmark;

namespace {

template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;

static auto kReduce = [](float* x, const float* y, size_t n) {
  EigenVectorMap<float>(x, n) =
      ConstEigenVectorMap<float>(x, n) + ConstEigenVectorMap<float>(y, n);
};

class Benchmark {
 public:
  Benchmark(std::shared_ptr<Context>& context, struct options& options)
      : context_(context), options_(options) {}

  virtual ~Benchmark() {}

  virtual void initialize(int elements) = 0;

  virtual void run() {
    algorithm_->Run();
  }

  virtual bool verify() = 0;

 protected:
  virtual float* allocate(int elements) {
    data_.resize(elements);
    for (int i = 0; i < data_.size(); i++) {
      data_[i] = context_->rank_;
    }
    return data_.data();
  }

  std::shared_ptr<Context> context_;
  struct options options_;
  std::unique_ptr<Algorithm> algorithm_;
  std::vector<float> data_;
};

class AllreduceRingBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int elements) override {
    auto ptr = allocate(elements);
    algorithm_.reset(
        new AllreduceRing<float>(context_, {ptr}, elements, kReduce));
  }

  virtual bool verify() override {
    auto expected = (context_->size_ * (context_->size_ - 1)) / 2;
    for (int i = 0; i < data_.size(); i++) {
      FBC_ENFORCE_EQ(expected, data_[i], "Mismatch at index ", i);
    }
    return true;
  }
};

class AllreduceRingChunkedBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int elements) override {
    auto ptr = allocate(elements);
    algorithm_.reset(
        new AllreduceRingChunked<float>(context_, {ptr}, elements, kReduce));
  }

  virtual bool verify() override {
    auto expected = (context_->size_ * (context_->size_ - 1)) / 2;
    for (int i = 0; i < data_.size(); i++) {
      FBC_ENFORCE_EQ(expected, data_[i], "Mismatch at index ", i);
    }
    return true;
  }
};

class BarrierAllToAllBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int /* unused */) override {
    algorithm_.reset(new BarrierAllToAll(context_));
  }

  virtual bool verify() override {
    return true;
  }
};

class BarrierAllToOneBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int /* unused */) override {
    // This tool measures at rank=0, so use root=1 for the all to one
    // barrier to measure the end-to-end latency (otherwise we might
    // not account for the send-to-root part of the algorithm).
    algorithm_.reset(new BarrierAllToOne(context_, 1));
  }

  virtual bool verify() override {
    return true;
  }
};

class BroadcastOneToAllBenchmark : public Benchmark {
  using Benchmark::Benchmark;

 public:
  virtual void initialize(int elements) override {
    auto ptr = allocate(elements);
    algorithm_.reset(
        new BroadcastOneToAll<float>(context_, ptr, elements, rootRank_));
  }

  virtual bool verify() override {
    for (int i = 0; i < data_.size(); i++) {
      FBC_ENFORCE_EQ(rootRank_, data_[i], "Mismatch at index ", i);
    }
    return true;
  }

 protected:
  const int rootRank_ = 0;
};

class Runner {
 public:
  using BenchmarkFn =
      std::function<std::unique_ptr<Benchmark>(std::shared_ptr<Context>&)>;

  explicit Runner(const options& options) : options_(options) {
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
    FBC_ENFORCE(device_, "Unknown transport: ", options_.transport);

    // Create broadcast algorithm to synchronize between participants
    broadcast_.reset(
        new BroadcastOneToAll<long>(newContext(), &broadcastValue_, 1));
    // Create barrier for run-to-run synchronization
    barrier_.reset(new BarrierAllToOne(newContext()));
  }

  long broadcast(long value) {
    broadcastValue_ = value;
    broadcast_->Run();
    return broadcastValue_;
  }

  std::string newPrefix() {
    std::stringstream prefix;
    prefix << options_.prefix << "-" << prefixCounter_++;
    return prefix.str();
  }

  std::shared_ptr<Context> newContext() {
    auto context =
        std::make_shared<Context>(options_.contextRank, options_.contextSize);
    auto redisStore = std::unique_ptr<rendezvous::Store>(
        new rendezvous::RedisStore(options_.redisHost, options_.redisPort));
    auto prefixStore = std::unique_ptr<rendezvous::Store>(
        new rendezvous::PrefixStore(newPrefix(), redisStore));
    context->connectFullMesh(*prefixStore, device_);
    return context;
  }

  void run(BenchmarkFn& fn) {
    printHeader();

    if (options_.elements > 0) {
      run(fn, options_.elements);
      return;
    }

    // Run sweep over number of elements
    for (int i = 1; i <= 1000000; i *= 10) {
      std::vector<int> js = {i * 1, i * 2, i * 5};
      for (auto& j : js) {
        run(fn, j);
      }
    }
  }

  void run(BenchmarkFn& fn, int n) {
    auto context = newContext();
    auto benchmark = fn(context);
    benchmark->initialize(n);

    // Verify correctness of initial run
    if (options_.verify) {
      benchmark->run();
      FBC_ENFORCE(benchmark->verify());
    }

    // Switch mode based on iteration count or time spent
    auto iterations = options_.iterationCount;
    if (iterations <= 0) {
      FBC_ENFORCE_GT(options_.iterationTimeNanos, 0);

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
    d_.clear();
    for (int i = 0; i < iterations; i++) {
      Timer dt;
      benchmark->run();
      d_.add(dt);
    }

    printDistribution(n);

    // Barrier to make sure everybody arrived here and the temporary
    // context and benchmark can be destructed.
    barrier_->Run();
  }

  void printHeader() {
    if (options_.contextRank == 0) {
      std::cout << std::setw(11) << "elements" << std::setw(11) << "min (us)"
                << std::setw(11) << "p50 (us)" << std::setw(11) << "p99 (us)"
                << std::setw(11) << "max (us)" << std::setw(11) << "samples"
                << std::endl;
    }
  }

  void printDistribution(int elements) {
    if (options_.contextRank == 0) {
      std::cout << std::setw(11) << elements << std::setw(11)
                << d_.percentile(0.00) / 1000 << std::setw(11)
                << d_.percentile(0.50) / 1000 << std::setw(11)
                << d_.percentile(0.90) / 1000 << std::setw(11)
                << d_.percentile(0.99) / 1000 << std::setw(11) << d_.size()
                << std::endl;
    }
  }

 protected:
  options options_;
  int prefixCounter_ = 0;
  std::shared_ptr<transport::Device> device_;

  long broadcastValue_;
  std::unique_ptr<Algorithm> broadcast_;
  std::unique_ptr<Algorithm> barrier_;

  Distribution d_;
};

} // namespace

// make_unique is a C++14 feature. If we don't have 14, we will emulate
// its behavior. This is copied from folly/Memory.h
#if __cplusplus >= 201402L ||                                              \
    (defined __cpp_lib_make_unique && __cpp_lib_make_unique >= 201304L) || \
    (defined(_MSC_VER) && _MSC_VER >= 1900)
/* using override */ using std::make_unique;
#else

template <typename T, typename... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Allows 'make_unique<T[]>(10)'. (N3690 s20.9.1.4 p3-4)
template <typename T>
typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(const size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

// Disallows 'make_unique<T[10]>()'. (N3690 s20.9.1.4 p5)
template <typename T, typename... Args>
typename std::enable_if<std::extent<T>::value != 0, std::unique_ptr<T>>::type
make_unique(Args&&...) = delete;

#endif

int main(int argc, char** argv) {
  auto x = benchmark::parseOptions(argc, argv);

  Runner::BenchmarkFn fn;
  if (x.benchmark == "allreduce_ring") {
    fn = [&](std::shared_ptr<Context>& context) {
      return make_unique<AllreduceRingBenchmark>(context, x);
    };
  } else if (x.benchmark == "allreduce_ring_chunked") {
    fn = [&](std::shared_ptr<Context>& context) {
      return make_unique<AllreduceRingChunkedBenchmark>(context, x);
    };
  } else if (x.benchmark == "barrier_all_to_all") {
    fn = [&](std::shared_ptr<Context>& context) {
      return make_unique<BarrierAllToAllBenchmark>(context, x);
    };
  } else if (x.benchmark == "barrier_all_to_one") {
    fn = [&](std::shared_ptr<Context>& context) {
      return make_unique<BarrierAllToOneBenchmark>(context, x);
    };
  } else if (x.benchmark == "broadcast_one_to_all") {
    fn = [&](std::shared_ptr<Context>& context) {
      return make_unique<BroadcastOneToAllBenchmark>(context, x);
    };
  }

  if (!fn) {
    FBC_ENFORCE(false, "Invalid algorithm: ", x.benchmark);
  }

  Runner r(x);
  r.run(fn);
  return 0;
}
