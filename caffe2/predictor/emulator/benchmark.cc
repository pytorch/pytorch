#include "caffe2/core/init.h"
#include "caffe2/predictor/emulator/std_output_formatter.h"

#include "benchmark.h"

// Basic benchmark params
C10_DEFINE_int(warmup, 10000, "The number of iterations to warm up.");
C10_DEFINE_int(iter, 10000000, "The number of iterations to run.");
C10_DEFINE_int(threads, 32, "The number of threads to run.");
C10_DEFINE_int(runs, 10, "The number of independent runs.");

// Benchmark setup params
C10_DEFINE_int(
    num_loading_threads,
    56,
    "The number of threads to build predictors.");

// Benchmark model params
C10_DEFINE_string(run_net, "", "The given net to benchmark.");
C10_DEFINE_string(init_net, "", "The given net to initialize.");
C10_DEFINE_string(data_net, "", "The given net to get input data.");
C10_DEFINE_string(
    input_dims,
    "",
    "The path of the file that "
    "stores input dimensions of all the operators in the run net. "
    "Each element of the array is a mapping from "
    "operator index to its input dimension.");
C10_DEFINE_string(
    input_types,
    "",
    "The path of the file that "
    "stores input types of all the operators in the run net. "
    "Each element of the array is a mapping from "
    "operator index to its input types.");

namespace caffe2 {
namespace emulator {

void BenchmarkRunner::benchmark(const BenchmarkParam& param) {
  param.emulator->init();
  std::vector<float> durations_ms;
  for (size_t run = 0; run < FLAGS_runs; ++run) {
    LOG(WARNING) << "Starting run " << run + 1;
    LOG(INFO) << "Warming up " << FLAGS_threads << " threads with "
              << FLAGS_warmup << " iterations...";
    param.emulator->run(FLAGS_warmup);

    LOG(INFO) << "Starting benchmark with " << FLAGS_iter << " iterations...";
    pre_benchmark_setup();
    const auto duration_ms =
        param.profiler->profile([&]() { param.emulator->run(FLAGS_iter); });

    durations_ms.emplace_back(duration_ms);
    auto throughput = FLAGS_iter / (duration_ms / MS_IN_SECOND);
    LOG(INFO) << "Benchmark run finished in " << duration_ms / MS_IN_SECOND
              << " seconds.\n"
              << "Throughput:\t\t" << throughput << " iterations/s\n";

    post_benchmark_cleanup();
    LOG(INFO) << "Run " << run + 1 << " finished";
  }
  LOG(WARNING) << param.formatter->format(
      durations_ms, FLAGS_threads, FLAGS_iter);
}

} // namespace emulator
} // namespace caffe2
