#pragma once

#include <random>
#include <thread>

#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace torch {
namespace throughput_benchmark {
namespace detail {

template <class Input, class Output, class Model>
BenchmarkExecutionStats BenchmarkHelper<Input, Output, Model>::benchmark(
    const BenchmarkConfig& config) const {
  CHECK(initialized_);
  TORCH_CHECK(
      config.num_worker_threads == 1,
      "Only parallelization by callers is supported");

  // We pre-generate inputs here for each of the threads. This allows us to
  // safely move inputs out for each of the threads independently and thus avoid
  // overhead from the benchmark runner itself
  std::vector<std::vector<Input>> thread_inputs(config.num_calling_threads);
  std::vector<size_t> input_iters(config.num_calling_threads);
  {
    std::random_device seeder;
    std::mt19937 engine(seeder());
    TORCH_CHECK(
        !inputs_.empty(),
        "Please provide benchmark inptus."
        "Did you forget to call add_input()? ");
    std::uniform_int_distribution<int> dist(0, inputs_.size() - 1);

    for (int thread_id = 0; thread_id < config.num_calling_threads;
         ++thread_id) {
      // Just in case we generate num_iters inputs for each of the threads
      // This was if one thread does all the work we will be fine
      for (int i = 0; i < config.num_iters + config.num_warmup_iters; ++i) {
        thread_inputs[thread_id].push_back(cloneInput(inputs_[dist(engine)]));
      }
      input_iters[thread_id] = 0;
    }
  }

  std::mutex m;
  std::condition_variable worker_main_cv;
  std::condition_variable main_worker_cv;
  // TODO: add GUARDED_BY once it is available
  int64_t initialized{0};
  int64_t finished{0};
  bool start{false};
  std::atomic<int64_t> num_attempted_iters{0};
  std::vector<std::thread> callers;

  for (auto thread_id = 0; thread_id < config.num_calling_threads;
       ++thread_id) {
    callers.emplace_back([&, thread_id]() {
      // We use conditional variable as a barrier to make sure each thread
      // performs required warmeup iterations before we start measuring
      for (auto j = 0; j < config.num_warmup_iters; ++j) {
        runOnce(std::move(thread_inputs[thread_id][input_iters[thread_id]]));
        ++input_iters[thread_id];
      }
      {
        std::unique_lock<std::mutex> lock(m);
        ++initialized;
        worker_main_cv.notify_one();
        while (!start) {
          main_worker_cv.wait(lock);
        }
      }
      LOG(INFO) << "Starting forward thread " << thread_id;
      while (num_attempted_iters.fetch_add(1) < config.num_iters) {
        runOnce(std::move(thread_inputs[thread_id][input_iters[thread_id]]));
        ++input_iters[thread_id];
      }

      {
        std::unique_lock<std::mutex> lock(m);
        ++finished;
        worker_main_cv.notify_one();
        LOG(INFO) << "Shutting down forward thread " << thread_id
                  << ". Total number of finished threads: " << finished;
      }

    });
  }

  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;
  TimePoint start_time;

  {
    std::unique_lock<std::mutex> lock(m);
    while (initialized != config.num_calling_threads) {
      worker_main_cv.wait(lock);
    }
    LOG(INFO) << "Starting threads";
    start = true;
    start_time = Clock::now();
  }

  main_worker_cv.notify_all();
  {
    std::unique_lock<std::mutex> lock(m);
    worker_main_cv.wait(
        lock, [&]() { return finished == config.num_calling_threads; });
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Finished benchmark";

  BenchmarkExecutionStats stats;
  float total_time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            end_time - start_time)
                            .count() /
      1000.0 / 1000.0;
  // We use config.num_iters instead of num_attempted_iters as it is
  // repsesatative of the real work done. Last attempted iteration on each
  // calling threads doesn't represent the real work (i.e. running the model)
  stats.latency_avg_ms =
      total_time_ms * config.num_calling_threads / config.num_iters;
  stats.num_iters = config.num_iters;

  for (auto& t : callers) {
    t.join();
  }
  return stats;
}

} // namespace detail
} // namespace throughput_benchmark
} // namespace torch
