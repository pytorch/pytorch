#pragma once

#include <random>
#include <thread>

#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/Parallel.h>
#include <ATen/autocast_mode.h>
#include <c10/core/GradMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/irange.h>

namespace torch::throughput_benchmark::detail {

template <class Input, class Output, class Model>
BenchmarkExecutionStats BenchmarkHelper<Input, Output, Model>::benchmark(
    const BenchmarkConfig& config) const {
  CHECK(initialized_);
  TORCH_CHECK(
      config.num_worker_threads == 1,
      "Only parallelization by callers is supported");

  LOG(INFO) << at::get_parallel_info();

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
        "Please provide benchmark inputs."
        "Did you forget to call add_input()? ");
    std::uniform_int_distribution<int> dist(0, inputs_.size() - 1);

    for (const auto thread_id : c10::irange(config.num_calling_threads)) {
      // Just in case we generate num_iters inputs for each of the threads
      // This was if one thread does all the work we will be fine
      for (const auto i [[maybe_unused]] :
           c10::irange(config.num_iters + config.num_warmup_iters)) {
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

  callers.reserve(config.num_calling_threads);

  static constexpr auto& DEVICES = at::autocast::_AUTOCAST_SUPPORTED_DEVICES;
  std::array<bool, DEVICES.size()> autocast_enabled;
  std::array<at::ScalarType, DEVICES.size()> autocast_dtype;
  for (size_t i = 0; i < DEVICES.size(); i++) {
    autocast_enabled[i] = at::autocast::is_autocast_enabled(DEVICES[i]);
    autocast_dtype[i] = at::autocast::get_autocast_dtype(DEVICES[i]);
  }
  bool autocast_cache_enabled = at::autocast::is_autocast_cache_enabled();
  bool tls_grad_enabled = c10::GradMode::is_enabled();
  c10::impl::LocalDispatchKeySet tls_key_set =
      c10::impl::tls_local_dispatch_key_set();

  for (const auto thread_id : c10::irange(config.num_calling_threads)) {
    callers.emplace_back([&, thread_id]() {
      // We use conditional variable as a barrier to make sure each thread
      // performs required warmeup iterations before we start measuring
      c10::GradMode::set_enabled(tls_grad_enabled);
      c10::impl::_force_tls_local_dispatch_key_set(tls_key_set);
      for (size_t i = 0; i < DEVICES.size(); i++) {
        at::autocast::set_autocast_enabled(DEVICES[i], autocast_enabled[i]);
        at::autocast::set_autocast_dtype(DEVICES[i], autocast_dtype[i]);
      }
      at::autocast::set_autocast_cache_enabled(autocast_cache_enabled);

      for (const auto j : c10::irange(config.num_warmup_iters)) {
        (void)j;
        runOnce(std::move(thread_inputs[thread_id][input_iters[thread_id]]));
        ++input_iters[thread_id];
      }
      {
        std::unique_lock<std::mutex> lock(m);
        ++initialized;
        worker_main_cv.notify_one();
        // NOLINTNEXTLINE(bugprone-infinite-loop)
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
  using RecordProfile = torch::autograd::profiler::RecordProfile;
  using TimePoint = std::chrono::time_point<Clock>;
  TimePoint start_time;

  std::unique_ptr<RecordProfile> profiler_guard;
  {
    std::unique_lock<std::mutex> lock(m);
    while (initialized != config.num_calling_threads) {
      worker_main_cv.wait(lock);
    }
    if (!config.profiler_output_path.empty()) {
      LOG(INFO) << "Using Autograd profiler. Trace will be saved to "
                << config.profiler_output_path;
      profiler_guard =
          std::make_unique<RecordProfile>(config.profiler_output_path);
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
  profiler_guard.reset();
  LOG(INFO) << "Finished benchmark";

  BenchmarkExecutionStats stats;
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  float total_time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            end_time - start_time)
                            .count() /
      1000.0 / 1000.0;
  // We use config.num_iters instead of num_attempted_iters as it is
  // repsesatative of the real work done. Last attempted iteration on each
  // calling threads doesn't represent the real work (i.e. running the model)
  stats.latency_avg_ms =
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      total_time_ms * config.num_calling_threads / config.num_iters;
  stats.num_iters = config.num_iters;

  for (auto& t : callers) {
    t.join();
  }
  return stats;
}

} // namespace torch::throughput_benchmark::detail
