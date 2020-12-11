#include <torch/torch.h>
#include <ATen/record_function.h>

#include "c10/util/Flags.h"

#include <chrono>
#include <iostream>
#include <ctime>

C10_DEFINE_int(iter, 100, "Number of iterations");
C10_DEFINE_int(warmup_iter, 10, "Number of warmup iterations");
C10_DEFINE_int(sampled_iter, 10e6,
    "Number of iterations for the sampled observer benchmark");

namespace {
const int kInnerIter = 100;
const int kNumSampledCb = 2;
const int kTensorSize = 16;
const int kSmallTensorSize = 1;
const float kSampingProb = 0.1;

const float kLowSamplingProb = 0.0001;
}

void setupBenchmarkCallbacks() {
  at::enableRecordFunction();
  at::clearCallbacks();
  // non-sampled callback
  at::addGlobalCallback(at::RecordFunctionCallback(
      [&](const at::RecordFunction& fn) {},
      [](const at::RecordFunction&) {})
    .needsInputs(true));

  // sampled
  for (auto idx = 0; idx < kNumSampledCb; ++idx) {
    at::addGlobalCallback(at::RecordFunctionCallback(
        [](const at::RecordFunction& fn) {},
        [](const at::RecordFunction&) {})
      .needsInputs(true)
      .samplingProb(kSampingProb)
    );
  }
}

float runTensorBench(int tensor_size, int outer_iter) {
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::microseconds us;
  std::chrono::time_point<clock> start_time = clock::now();
  for (auto idx = 0; idx < kInnerIter * outer_iter; ++idx) {
    torch::mm(
        torch::randn({tensor_size, tensor_size}),
        torch::randn({tensor_size, tensor_size}));
  }
  auto duration = static_cast<float>(
      std::chrono::duration_cast<us>(clock::now() - start_time).count());
  return duration;
}

float runPureRecordFunctionBench(int outer_iter) {
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::microseconds us;
  std::chrono::time_point<clock> start_time = clock::now();
  for (auto n = 0; n < outer_iter; ++n) {
    RECORD_USER_SCOPE("test");
  }
  auto duration = static_cast<float>(
      std::chrono::duration_cast<us>(clock::now() - start_time).count());
  return duration;
}

void runBenchmark() {
  float duration = 0;
  for (auto tensor_size : std::set<int>({kSmallTensorSize, kTensorSize})) {
    duration = runTensorBench(tensor_size, FLAGS_iter);
    std::cout << "Running tensor benchmark, time per iteration ("
              << tensor_size
              << "x"
              << tensor_size
              << "): " << (duration/FLAGS_iter)
              << " us." << std::endl;
  }
  duration = runPureRecordFunctionBench(FLAGS_iter * 100);
  std::cout << "Running pure RecordFunction benchmark, time per iteration: "
              << (duration/FLAGS_iter)
              << " us." << std::endl;
}

int main(int argc, char** argv) {
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cout << "Failed to parse command line flags" << std::endl;
    return -1;
  }

  auto duration = runTensorBench(kSmallTensorSize, FLAGS_warmup_iter);
  std::cout << "Warmup time: " << duration << " us." << std::endl;

  setupBenchmarkCallbacks();
  std::cout << "Running with empty observers" << std::endl;
  runBenchmark();

  at::clearCallbacks();
  std::cout << "Running without observers" << std::endl;
  runBenchmark();

  std::cout << "Running sampled observer benchmark" << std::endl;
  int cb_count = 0;
  at::addGlobalCallback(at::RecordFunctionCallback(
      [&](const at::RecordFunction& fn) {
        ++cb_count;
      },
      [](const at::RecordFunction&) {})
    .needsInputs(true)
    .samplingProb(kLowSamplingProb)
  );

  runPureRecordFunctionBench(FLAGS_sampled_iter);

  std::cout << "Pure RecordFunction runtime of " << FLAGS_sampled_iter
            << " iterations " << duration
            << " us, number of callback invocations: " << cb_count
            << ", expected number: ~" << (int)(FLAGS_sampled_iter * kLowSamplingProb)
            << " invocations" << std::endl;

  at::clearCallbacks();
  return 0;
}
