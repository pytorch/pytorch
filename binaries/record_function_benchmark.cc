#include <torch/torch.h>
#include <ATen/record_function.h>

#include "c10/util/Flags.h"

#include <chrono>
#include <iostream>
#include <ctime>

C10_DEFINE_int(iter, 100, "Number of iterations");
C10_DEFINE_int(warmup_iter, 10, "Number of warmup iterations");
C10_DEFINE_int(rec_fn_iter, 10e6,
    "Number of iterations for the pure RecordFunction benchmark");

namespace {
const int kInnerIter = 100;
const int kNumSampledCb = 2;
const int kTensorSize = 16;
const int kSmallTensorSize = 1;
const float kSampingProb = 0.1;

const float kLowSamplingProb = 0.0001;
}

void setupBenchmarkCallbacks() {
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

float runBench(int tensor_size, int outer_iter) {
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

int main(int argc, char** argv) {
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cout << "Failed to parse command line flags" << std::endl;
    return -1;
  }

  at::enableRecordFunction();
  setupBenchmarkCallbacks();

  auto duration = runBench(kSmallTensorSize, FLAGS_warmup_iter);
  std::cout << "Warmup time: " << duration << " us." << std::endl;

  for (auto tensor_size : std::set<int>({kSmallTensorSize, kTensorSize})) {
    duration = runBench(tensor_size, FLAGS_iter);
    std::cout << "Time per iteration ("
              << tensor_size
              << "x"
              << tensor_size
              << "): " << (duration/FLAGS_iter)
              << " us." << std::endl;
  }

  at::clearCallbacks();

  int cb_count = 0;
  at::addGlobalCallback(at::RecordFunctionCallback(
      [&](const at::RecordFunction& fn) {
        ++cb_count;
      },
      [](const at::RecordFunction&) {})
    .needsInputs(true)
    .samplingProb(kLowSamplingProb)
  );

  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::microseconds us;
  std::chrono::time_point<clock> start_time = clock::now();
  for (auto n = 0; n < FLAGS_rec_fn_iter; ++n) {
    RECORD_USER_SCOPE("test");
  }
  duration = static_cast<float>(
      std::chrono::duration_cast<us>(clock::now() - start_time).count());
  std::cout << "Pure RecordFunction runtime of " << FLAGS_rec_fn_iter
            << " iterations " << duration
            << " us, number of callback invocations: " << cb_count
            << ", expected number: ~" << (int)(FLAGS_rec_fn_iter * kLowSamplingProb)
            << " invocations" << std::endl;

  at::clearCallbacks();
  return 0;
}
