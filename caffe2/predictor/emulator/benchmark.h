#pragma once
#include "caffe2/core/logging.h"
#include "caffe2/predictor/emulator/emulator.h"
#include "caffe2/predictor/emulator/output_formatter.h"
#include "caffe2/predictor/emulator/profiler.h"

C10_DECLARE_int(warmup);
C10_DECLARE_int(iter);
C10_DECLARE_int(threads);
C10_DECLARE_int(runs);
C10_DECLARE_string(run_net);
C10_DECLARE_string(init_net);
C10_DECLARE_string(data_net);
C10_DECLARE_string(input_dims);
C10_DECLARE_string(input_types);

namespace caffe2 {
namespace emulator {

struct BenchmarkParam {
  std::unique_ptr<Profiler> profiler;
  std::unique_ptr<Emulator> emulator;
  std::unique_ptr<OutputFormatter> formatter;
};

/*
 * benchmark runner takes an @emulator to run nets.
 * The runtime will be measured by @profiler.
 * The output will be formatted by @formatter
 */
class BenchmarkRunner {
 public:
  void benchmark(const BenchmarkParam& param);

  virtual ~BenchmarkRunner() noexcept {}

 protected:
  virtual void pre_benchmark_setup() {}

  virtual void post_benchmark_cleanup() {}
};

} // namespace emulator
} // namespace caffe2
