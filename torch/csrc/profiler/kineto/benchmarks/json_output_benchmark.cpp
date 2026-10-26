/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Benchmark for measuring JSON output file writing performance in Kineto.
// Tests small (<1KB), medium (~1MB), and large (~1GB) JSON file scenarios.
//
// CMake usage:
//   mkdir build && cd build
//   cmake .. -DKINETO_BUILD_BENCHMARKS=ON
//   make json_output_benchmark
//   ./benchmarks/json_output_benchmark --scenario=all --output_dir=/tmp

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>
#include <ratio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include "ActivityType.h"
#include "Config.h"
#include "GenericTraceActivity.h"
#include "TraceSpan.h"
#include "output_json.h"
#include "time_since_epoch.h"

namespace {

using namespace libkineto;

// Command line argument defaults
struct BenchmarkOptions {
  std::string scenario = "all";
  std::string output_dir = "/tmp";
  int small_iterations = 100;
  int medium_iterations = 20;
  int large_iterations = 5;
  bool keep_files = false;
};

void printUsage(const char* progname) {
  fmt::print("Usage: {} [options]\n", progname);
  fmt::print("Options:\n");
  fmt::print(
      "  --scenario=<small|medium|large|all>  Scenario to run (default: all)\n");
  fmt::print(
      "  --output_dir=<path>                  Output directory (default: /tmp)\n");
  fmt::print(
      "  --small_iterations=<n>               Iterations for small (default: 100)\n");
  fmt::print(
      "  --medium_iterations=<n>              Iterations for medium (default: 20)\n");
  fmt::print(
      "  --large_iterations=<n>               Iterations for large (default: 5)\n");
  fmt::print(
      "  --keep_files                         Keep generated JSON files\n");
  fmt::print("  --help                               Show this help\n");
}

BenchmarkOptions parseArgs(int argc, char* argv[]) {
  BenchmarkOptions opts;
  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
    if (arg.starts_with("--scenario=")) {
      opts.scenario = arg.substr(11);
    } else if (arg.starts_with("--output_dir=")) {
      opts.output_dir = arg.substr(13);
    } else if (arg.starts_with("--small_iterations=")) {
      opts.small_iterations = std::stoi(arg.substr(19));
    } else if (arg.starts_with("--medium_iterations=")) {
      opts.medium_iterations = std::stoi(arg.substr(20));
    } else if (arg.starts_with("--large_iterations=")) {
      opts.large_iterations = std::stoi(arg.substr(19));
    } else if (arg == "--keep_files") {
      opts.keep_files = true;
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    }
  }
  return opts;
}

// Realistic operation names for different activity types
const std::vector<std::string> kCpuOpNames = {
    "aten::conv2d",
    "aten::linear",
    "aten::relu",
    "aten::batch_norm",
    "aten::add",
    "aten::mul",
    "aten::matmul",
    "aten::softmax",
    "aten::embedding",
    "aten::layer_norm",
    "aten::gelu",
    "aten::dropout",
};

const std::vector<std::string> kKernelNames = {
    "volta_sgemm_128x64_nn",
    "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>>",
    "cudnn::detail::implicit_convolve_sgemm<float, 128, 6, 7, 3, 3, 5>",
    "void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_256x128_32x3_nn_align4>",
    "ampere_sgemm_128x32_sliced1x4_tn",
    "void gemmSN_TN_kernel<float, 128, 16, 2, 4, 6, 7, false>",
    "void at::native::reduce_kernel<512, 1, at::native::ReduceOp>",
    "void at::native::batch_norm_collect_statistics_kernel<float>",
};

const std::vector<std::string> kMemcpyNames = {
    "Memcpy HtoD (Pageable -> Device)",
    "Memcpy DtoH (Device -> Pageable)",
    "Memcpy DtoD (Device -> Device)",
};

const std::vector<std::string> kCudaRuntimeNames = {
    "cudaLaunchKernel",
    "cudaMemcpyAsync",
    "cudaStreamSynchronize",
    "cudaMalloc",
    "cudaFree",
};

// Generate activities with realistic data
std::vector<GenericTraceActivity> generateActivities(
    const TraceSpan& span,
    size_t count,
    std::mt19937& rng) {
  std::vector<GenericTraceActivity> activities;
  activities.reserve(count);

  int64_t currentTime = span.startTime;
  int32_t correlationId = 1;

  std::uniform_int_distribution<> typeDist(0, 3);
  std::uniform_int_distribution<> durationDist(
      1000, 100000); // 1-100 microseconds in ns
  std::uniform_int_distribution<> gapDist(100, 1000); // 100ns - 1us gap

  for (size_t i = 0; i < count; ++i) {
    const int typeIdx = typeDist(rng);
    ActivityType type;
    std::string name;

    // Select activity type and name
    switch (typeIdx) {
      case 0:
        type = ActivityType::CPU_OP;
        name = kCpuOpNames[i % kCpuOpNames.size()];
        break;
      case 1:
        type = ActivityType::CONCURRENT_KERNEL;
        name = kKernelNames[i % kKernelNames.size()];
        break;
      case 2:
        type = ActivityType::GPU_MEMCPY;
        name = kMemcpyNames[i % kMemcpyNames.size()];
        break;
      case 3:
      default:
        type = ActivityType::CUDA_RUNTIME;
        name = kCudaRuntimeNames[i % kCudaRuntimeNames.size()];
        break;
    }

    GenericTraceActivity activity(span, type, name);
    activity.startTime = currentTime;
    activity.endTime = currentTime + durationDist(rng);
    activity.id = correlationId++;
    // CPU ops get process id as device, GPU ops get device 0
    activity.device = (typeIdx == 0) ? 12345 : 0;
    // Resource is thread id for CPU, stream id for GPU
    activity.resource = (typeIdx == 0) ? 67890 : static_cast<int32_t>(i % 8);
    activity.threadId = 67890;

    // Add metadata
    activity.addMetadata("External id", correlationId);
    if (typeIdx == 0) {
      // CPU ops get input shapes
      activity.addMetadataQuoted(
          "Input Dims", "[[32, 64, 28, 28], [64, 64, 3, 3]]");
      activity.addMetadata("Sequence number", static_cast<int64_t>(i));
    } else if (typeIdx == 1) {
      // GPU kernels get grid/block info
      activity.addMetadata("grid", 128);
      activity.addMetadata("block", 256);
      activity.addMetadata("shared memory", 49152);
      activity.addMetadata("registers per thread", 32);
    } else if (typeIdx == 2) {
      // Memcpy gets bytes
      activity.addMetadata("bytes", 1048576);
    }

    activities.push_back(std::move(activity));
    currentTime = activities.back().endTime + gapDist(rng);
  }

  return activities;
}

// Run a single benchmark iteration, returns time in milliseconds
double runBenchmarkIteration(
    const std::vector<GenericTraceActivity>& activities,
    const TraceSpan& span,
    const std::string& outputPath) {
  auto start = std::chrono::steady_clock::now();

  {
    ChromeTraceLogger logger(outputPath);

    // Initialize the trace with empty metadata
    std::unordered_map<std::string, std::string> metadata;
    logger.handleTraceStart(metadata, "");

    // Write all activities
    for (const auto& activity : activities) {
      logger.handleGenericActivity(activity);
    }

    // Finalize the trace
    const Config config;
    const int64_t endTime =
        activities.empty() ? span.endTime : activities.back().endTime;
    logger.finalizeTrace(config, nullptr, endTime);
  }

  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct BenchmarkStats {
  double mean_ms;
  double stddev_ms;
  double min_ms;
  double max_ms;
  size_t file_size_bytes;
};

BenchmarkStats calculateStats(
    const std::vector<double>& times,
    size_t fileSize) {
  BenchmarkStats stats{};
  stats.file_size_bytes = fileSize;

  if (times.empty()) {
    return stats;
  }

  const double sum = std::accumulate(times.begin(), times.end(), 0.0);
  stats.mean_ms = sum / static_cast<double>(times.size());

  double sqSum = 0.0;
  for (const double t : times) {
    sqSum += (t - stats.mean_ms) * (t - stats.mean_ms);
  }

  stats.stddev_ms = std::sqrt(sqSum / static_cast<double>(times.size()));
  stats.min_ms = *std::ranges::min_element(times);
  stats.max_ms = *std::ranges::max_element(times);

  return stats;
}

void printResults(
    const std::string& scenario,
    const BenchmarkStats& stats,
    int iterations,
    size_t activityCount) {
  double sizeMB =
      static_cast<double>(stats.file_size_bytes) / (1024.0 * 1024.0);
  double throughputMBps = sizeMB / (stats.mean_ms / 1000.0);

  fmt::print("\n");
  fmt::print("=== {} Scenario ===\n", scenario);
  fmt::print("Activities:  {}\n", activityCount);
  fmt::print(
      "File size:   {} bytes ({:.2f} MB)\n", stats.file_size_bytes, sizeMB);
  fmt::print("Iterations:  {}\n", iterations);
  fmt::print("Mean time:   {:.3f} ms\n", stats.mean_ms);
  fmt::print("Std dev:     {:.3f} ms\n", stats.stddev_ms);
  fmt::print("Min time:    {:.3f} ms\n", stats.min_ms);
  fmt::print("Max time:    {:.3f} ms\n", stats.max_ms);
  fmt::print("Throughput:  {:.2f} MB/s\n", throughputMBps);
}

void runScenario(
    const std::string& name,
    size_t activityCount,
    int iterations,
    const std::string& outputDir,
    bool keepFiles) {
  fmt::print(
      "Running {} scenario ({} activities, {} iterations)...\n",
      name,
      activityCount,
      iterations);

  // Fixed seed for reproducibility
  std::mt19937 rng(42);

  // Create trace span
  const int64_t baseTime = timeSinceEpoch(std::chrono::system_clock::now());
  const TraceSpan span(baseTime, baseTime + 1000000000LL, name + "Trace");

  // Generate activities once (not included in timing)
  auto activities = generateActivities(span, activityCount, rng);

  std::string outputPath = outputDir + "/benchmark_" + name + ".json";
  std::vector<double> times;
  times.reserve(iterations);

  for (int i = 0; i < iterations; ++i) {
    times.push_back(runBenchmarkIteration(activities, span, outputPath));
  }

  // Get file size from last iteration
  size_t fileSize = 0;
  try {
    fileSize = std::filesystem::file_size(outputPath);
  } catch (const std::filesystem::filesystem_error& e) {
    fmt::print("Warning: Could not get file size: {}\n", e.what());
  }

  auto stats = calculateStats(times, fileSize);
  printResults(name, stats, iterations, activityCount);

  if (!keepFiles) {
    std::filesystem::remove(outputPath);
  } else {
    fmt::print("Output file kept at: {}\n", outputPath);
  }
}

} // namespace

int main(int argc, char* argv[]) {
  auto opts = parseArgs(argc, argv);

  // Initialize ChromeTraceBaseTime singleton
  ChromeTraceBaseTime::singleton().init();

  fmt::print("Output directory: {}\n", opts.output_dir);

  const bool runSmall = opts.scenario == "all" || opts.scenario == "small";
  const bool runMedium = opts.scenario == "all" || opts.scenario == "medium";
  const bool runLarge = opts.scenario == "all" || opts.scenario == "large";

  // Small: ~5 activities, targeting <1KB
  if (runSmall) {
    runScenario(
        "small", 5, opts.small_iterations, opts.output_dir, opts.keep_files);
  }

  // Medium: ~6000 activities, targeting ~1MB
  if (runMedium) {
    runScenario(
        "medium",
        6000,
        opts.medium_iterations,
        opts.output_dir,
        opts.keep_files);
  }

  // Large: ~6M activities, targeting ~1GB
  if (runLarge) {
    fmt::print(
        "\nNote: Large scenario requires ~2GB RAM for activity generation.\n");
    runScenario(
        "large",
        6000000,
        opts.large_iterations,
        opts.output_dir,
        opts.keep_files);
  }

  return 0;
}
